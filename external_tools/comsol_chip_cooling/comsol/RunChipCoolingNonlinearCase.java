import com.comsol.model.Model;
import com.comsol.model.MeshFeature;
import com.comsol.model.util.ModelUtil;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Solve one transient conjugate chip-cooling case from the COMSOL 6.4
 * Application Library model. The installed model is opened as a copy and is
 * never overwritten.
 */
public final class RunChipCoolingNonlinearCase {
    private static final int BASE_DOMAIN = 2;
    private static final int[] FIN_DOMAINS = {3, 5, 6, 7};
    private static final int MAX_TRANSIENT_SEGREGATED_ITERATIONS = 25;

    private static final List<String> SCHEDULE_TIME = new ArrayList<>();
    private static final List<String> SCHEDULE_POWER = new ArrayList<>();
    private static final List<String> SCHEDULE_COOLANT_KELVIN = new ArrayList<>();
    private static final List<String> SCHEDULE_VELOCITY = new ArrayList<>();
    private static final List<String> SCHEDULE_HIDDEN_POWER = new ArrayList<>();
    private static final List<String> SCHEDULE_EFFECTIVE_VELOCITY = new ArrayList<>();
    private static boolean radiationEnabled;
    private static String activeMeshProfile;
    private static int activeBackgroundLevel;
    private static String[] activeLocalMeshValues;

    private RunChipCoolingNonlinearCase() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 5 && "steady-only".equals(args[0])) {
            readSchedule(args[2]);
            solveStationaryOnly(args[1], args[3], args[4]);
            return;
        }
        if (args.length == 5 && "mesh-only".equals(args[0])) {
            buildMeshOnly(args[1], args[2], args[3], args[4]);
            return;
        }
        if (args.length != 5) {
            throw new IllegalArgumentException(
                    "Usage: RunChipCoolingNonlinearCase <source.mph> <schedule-base64> "
                            + "<raw-output.txt> <edited-model.mph|-> <mesh-profile>\n"
                            + "   or: mesh-only <source.mph> <mesh-profile> "
                            + "<metrics.csv> <edited-model.mph|->");
        }
        String meshProfile = normalizeMeshProfile(args[4]);
        readSchedule(args[1]);

        Model model = ModelUtil.loadCopy("ChipCoolingNonlinearDataset", args[0]);
        Set<String> existingDatasets = new HashSet<>(Arrays.asList(
                model.result().dataset().tags()));
        configureModel(model, meshProfile);
        String initialSolutionTag = createStationaryStudy(model);

        System.out.println("Solving stationary initial field (" + initialSolutionTag + ")");
        model.sol(initialSolutionTag).runAll();
        activateTransientInputs(model);
        String solutionTag = createTransientStudy(model, initialSolutionTag);

        System.out.println("Solving " + (radiationEnabled ? "conjugate+radiation" : "conjugate")
                + " transient with " + SCHEDULE_TIME.size() + " output times, mesh profile "
                + meshProfile + " (" + solutionTag + ")");
        model.sol(solutionTag).runAll();

        String datasetTag = solutionDatasetTag(model, solutionTag, existingDatasets);
        exportResults(model, datasetTag, args[2], true);
        if (!"-".equals(args[3])) {
            model.save(args[3]);
            System.out.println("Saved edited model to " + args[3]);
        }
        System.out.println("Saved raw result table to " + args[2]);
        ModelUtil.remove("ChipCoolingNonlinearDataset");
    }

    private static void solveStationaryOnly(
            String source, String output, String meshProfile) throws Exception {
        Model model = ModelUtil.loadCopy("ChipCoolingMeshConvergence", source);
        Set<String> existingDatasets = new HashSet<>(Arrays.asList(
                model.result().dataset().tags()));
        configureModel(model, normalizeMeshProfile(meshProfile));
        String solutionTag = createStationaryStudy(model);
        System.out.println("Solving stationary mesh-convergence field (" + solutionTag + ")");
        model.sol(solutionTag).runAll();
        String datasetTag = solutionDatasetTag(model, solutionTag, existingDatasets);
        exportResults(model, datasetTag, output, false);
        System.out.println("Saved stationary result table to " + output);
        ModelUtil.remove("ChipCoolingMeshConvergence");
    }

    private static void buildMeshOnly(
            String source, String requestedProfile, String metricsOutput, String modelOutput)
            throws Exception {
        Model model = ModelUtil.loadCopy("ChipCoolingMeshInspection", source);
        configureMesh(model, normalizeMeshProfile(requestedProfile));
        printMeshMetricsCsv(model, metricsOutput);
        if (!"-".equals(modelOutput)) {
            model.save(modelOutput);
            System.out.println("Saved meshed model to " + modelOutput);
        }
        ModelUtil.remove("ChipCoolingMeshInspection");
    }

    private static void readSchedule(String encoded) {
        String csv = new String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8);
        String[] lines = csv.replace("\r", "").split("\n");
        if (lines.length < 2) {
            throw new IllegalArgumentException("Schedule is empty");
        }
        double previousTime = Double.NEGATIVE_INFINITY;
        Boolean firstRadiation = null;
        for (int i = 1; i < lines.length; i++) {
            if (lines[i].trim().isEmpty()) {
                continue;
            }
            String[] fields = lines[i].split(",", -1);
            if (fields.length != 7) {
                throw new IllegalArgumentException(
                        "Expected 7 schedule columns at line " + (i + 1));
            }
            double time = Double.parseDouble(fields[0].trim());
            double power = Double.parseDouble(fields[1].trim());
            double coolantCelsius = Double.parseDouble(fields[2].trim());
            double velocity = Double.parseDouble(fields[3].trim());
            double hiddenPower = Double.parseDouble(fields[4].trim());
            double effectiveVelocity = Double.parseDouble(fields[5].trim());
            boolean rowRadiation = Integer.parseInt(fields[6].trim()) != 0;
            if (!(time > previousTime)) {
                throw new IllegalArgumentException(
                        "Schedule time must be strictly increasing at line " + (i + 1));
            }
            if (power < 0.0 || hiddenPower < 0.0 || velocity <= 0.0
                    || effectiveVelocity <= 0.0) {
                throw new IllegalArgumentException(
                        "Power must be non-negative and velocity positive at line " + (i + 1));
            }
            if (firstRadiation != null && rowRadiation != firstRadiation) {
                throw new IllegalArgumentException("Radiation mode must be constant within a case");
            }
            firstRadiation = rowRadiation;
            SCHEDULE_TIME.add(Double.toString(time));
            SCHEDULE_POWER.add(Double.toString(power));
            SCHEDULE_COOLANT_KELVIN.add(Double.toString(coolantCelsius + 273.15));
            SCHEDULE_VELOCITY.add(Double.toString(velocity));
            SCHEDULE_HIDDEN_POWER.add(Double.toString(hiddenPower));
            SCHEDULE_EFFECTIVE_VELOCITY.add(Double.toString(effectiveVelocity));
            previousTime = time;
        }
        if (SCHEDULE_TIME.size() < 2 || Double.parseDouble(SCHEDULE_TIME.get(0)) != 0.0) {
            throw new IllegalArgumentException(
                    "Schedule must contain at least two rows and begin at time 0");
        }
        radiationEnabled = Boolean.TRUE.equals(firstRadiation);
    }

    private static void configureModel(Model model, String meshProfile) {
        System.out.println("Final named Chip domains: " + Arrays.toString(
                model.component("comp1").selection("sel1").entities(3)));

        model.component("comp1").physics("ht").feature("hs1")
                .set("P0", Double.toString(
                        Double.parseDouble(SCHEDULE_POWER.get(0))
                        + Double.parseDouble(SCHEDULE_HIDDEN_POWER.get(0))) + "[W]");
        model.component("comp1").common("ampr1")
                .set("T_amb", SCHEDULE_COOLANT_KELVIN.get(0) + "[K]");
        model.component("comp1").physics("spf").feature("inl1")
                .set("Uavfdf", SCHEDULE_EFFECTIVE_VELOCITY.get(0) + "[m/s]");
        model.component("comp1").physics("ht").feature("init1")
                .set("Tinit", SCHEDULE_COOLANT_KELVIN.get(0) + "[K]");
        if (radiationEnabled) {
            model.component("comp1").physics("rad").feature("dsurf1")
                    .set("Tamb", SCHEDULE_COOLANT_KELVIN.get(0) + "[K]");
            model.component("comp1").physics("rad").feature("init1")
                    .set("Tinit", SCHEDULE_COOLANT_KELVIN.get(0) + "[K]");
        }

        // Preserve the v1 solid properties so flow and radiation are the only
        // changed physical mechanisms in the paired comparison.
        model.component("comp1").material("mat2").label("Silicon (constant properties)");
        model.component("comp1").material("mat2").propertyGroup("def")
                .set("heatcapacity", "700[J/(kg*K)]");
        model.component("comp1").material("mat2").propertyGroup("def")
                .set("density", "2329[kg/m^3]");
        model.component("comp1").material("mat2").propertyGroup("def")
                .set("thermalconductivity", new String[]{
                        "148[W/(m*K)]", "0", "0",
                        "0", "148[W/(m*K)]", "0",
                        "0", "0", "148[W/(m*K)]"});

        createAverageNamed(model, "avg_chip_nl", "avg_chip", "sel1");
        createAverage(model, "avg_base_nl", "avg_base", new int[]{BASE_DOMAIN});
        createAverage(model, "avg_fins_nl", "avg_fins", FIN_DOMAINS);
        createMaximumNamed(model, "max_chip_nl", "max_chip", "sel1");
        createMaximum(model, "max_fins_nl", "max_fins", FIN_DOMAINS);
        createAverageBoundaryNamed(model, "avg_inlet_nl", "avg_inlet", "sel4");
        createAverageBoundaryNamed(model, "avg_outlet_nl", "avg_outlet", "sel5");
        createIntegrationBoundaryNamed(
                model, "int_rad_sink_nl", "int_rad_sink", "geom1_pi1_difsel1");
        createIntegrationNamed(model, "int_chip_nl", "int_chip", "sel1");
        createIntegration(model, "int_base_nl", "int_base", new int[]{BASE_DOMAIN});
        createIntegration(model, "int_fins_nl", "int_fins", FIN_DOMAINS);

        configureMesh(model, meshProfile);
    }

    private static String normalizeMeshProfile(String requested) {
        if (requested.matches("[1-9]")) {
            return "global-" + requested;
        }
        if (requested.matches("global-[1-9]")
                || "local-coarse".equals(requested)
                || "local-medium".equals(requested)
                || "local-fine".equals(requested)) {
            return requested;
        }
        throw new IllegalArgumentException("Unsupported mesh profile: " + requested);
    }

    private static void configureMesh(Model model, String profile) {
        activeMeshProfile = profile;
        activeLocalMeshValues = null;
        if (profile.startsWith("global-")) {
            int level = Integer.parseInt(profile.substring("global-".length()));
            activeBackgroundLevel = level;
            model.component("comp1").mesh("mesh1").autoMeshSize(level);
            model.component("comp1").mesh("mesh1").run();
            printMeshMetrics(model);
            return;
        }

        activeBackgroundLevel = 8;
        activeLocalMeshValues = localMeshValues(profile);
        model.component("comp1").mesh("mesh1").autoMeshSize(activeBackgroundLevel);
        model.component("comp1").mesh("mesh1").automatic(false);
        configureLocalSizes(model, activeLocalMeshValues);
        configureBoundaryLayers(model, activeLocalMeshValues);
        model.component("comp1").mesh("mesh1").run();
        printMeshMetrics(model);
    }

    private static String[] localMeshValues(String profile) {
        if ("local-coarse".equals(profile)) {
            return new String[]{
                    "10", "2.0", "1.2", "1.8", "0.8", "8",
                    "5", "1.5", "4", "2.5"};
        }
        if ("local-medium".equals(profile)) {
            return new String[]{
                    "7.5", "1.5", "0.9", "1.2", "0.6", "6",
                    "6", "1.5", "5", "2.5"};
        }
        if ("local-fine".equals(profile)) {
            return new String[]{
                    "6", "1.1", "0.65", "0.85", "0.4", "5",
                    "8", "1.6", "6", "2.5"};
        }
        throw new IllegalArgumentException("Unknown local mesh profile: " + profile);
    }

    private static void configureLocalSizes(Model model, String[] values) {
        MeshFeature freeTet = model.component("comp1").mesh("mesh1").feature("ftet1");
        addNamedSize(freeTet, "hf_air", 3, "sel3", millimeters(values[0]),
                "0.5[mm]", "1.35", "0.5", "1");
        addNamedSize(freeTet, "hf_solid", 3, "geom1_pi1_csel9_dom", millimeters(values[1]),
                "0.25[mm]", "1.25", "0.35", "1");
        addNamedSize(freeTet, "hf_chip", 3, "sel1", millimeters(values[2]),
                "0.2[mm]", "1.2", "0.3", "1");
        addNamedSize(freeTet, "hf_sink_wall", 2, "geom1_pi1_difsel1", millimeters(values[3]),
                "0.2[mm]", "1.2", "0.3", "1");
        addNamedSize(freeTet, "hf_contact", 2, "sel2", millimeters(values[4]),
                "0.15[mm]", "1.2", "0.3", "1");

        freeTet.feature().create("hf_channel_wall", "Size");
        MeshFeature channelWall = freeTet.feature("hf_channel_wall");
        channelWall.selection().geom("geom1", 2);
        channelWall.selection().set(1, 3, 4, 44);
        setSize(channelWall, millimeters(values[5]), "0.5[mm]", "1.3", "0.5", "1");
    }

    private static void configureBoundaryLayers(Model model, String[] values) {
        MeshFeature boundaryLayer = model.component("comp1").mesh("mesh1").feature("bl1");
        MeshFeature channel = boundaryLayer.feature("blp1");
        channel.selection().geom("geom1", 2);
        channel.selection().set(1, 3, 4, 44);
        setBoundaryLayer(channel, Integer.parseInt(values[8]), millimeters(values[9]));

        boundaryLayer.feature().create("hf_sink_bl", "BndLayerProp");
        MeshFeature sink = boundaryLayer.feature("hf_sink_bl");
        sink.selection().geom("geom1", 2);
        sink.selection().named("geom1_pi1_difsel1");
        setBoundaryLayer(sink, Integer.parseInt(values[6]), millimeters(values[7]));
    }

    private static String millimeters(String value) {
        return value + "[mm]";
    }

    private static void addNamedSize(
            MeshFeature parent,
            String tag,
            int dimension,
            String selection,
            String hmax,
            String hmin,
            String growth,
            String curvature,
            String narrow) {
        parent.feature().create(tag, "Size");
        MeshFeature size = parent.feature(tag);
        size.selection().geom("geom1", dimension);
        size.selection().named(selection);
        setSize(size, hmax, hmin, growth, curvature, narrow);
    }

    private static void setSize(
            MeshFeature size,
            String hmax,
            String hmin,
            String growth,
            String curvature,
            String narrow) {
        size.set("custom", "on");
        size.set("hmax", hmax);
        size.set("hmin", hmin);
        size.set("hgrad", growth);
        size.set("hcurve", curvature);
        size.set("hnarrow", narrow);
    }

    private static void setBoundaryLayer(MeshFeature feature, int layers, String thickness) {
        feature.set("blnlayers", layers);
        feature.set("blstretch", 1.2);
        feature.set("inittype", "blhtot");
        feature.set("blhtot", thickness);
    }

    private static void printMeshMetrics(Model model) {
        var statistics = model.component("comp1").mesh("mesh1").stat();
        statistics.setQualityMeasure("skewness");
        System.out.println("MESH profile=" + activeMeshProfile
                + " elements=" + statistics.getNumElem()
                + " tet=" + statistics.getNumElem("tet")
                + " prism=" + statistics.getNumElem("prism")
                + " min_quality=" + statistics.getMinQuality()
                + " mean_quality=" + statistics.getMeanQuality());
    }

    private static void printMeshMetricsCsv(Model model, String outputLabel) {
        var statistics = model.component("comp1").mesh("mesh1").stat();
        statistics.setQualityMeasure("skewness");
        String[] local = activeLocalMeshValues;
        String header = "mesh_profile,background_level,elements,tetrahedra,pyramids,prisms,"
                + "minimum_skewness_quality,mean_skewness_quality,air_hmax_mm,solid_hmax_mm,"
                + "chip_hmax_mm,sink_wall_hmax_mm,contact_hmax_mm,channel_wall_hmax_mm,"
                + "sink_boundary_layers,sink_boundary_layer_thickness_mm,"
                + "channel_boundary_layers,channel_boundary_layer_thickness_mm";
        String row = String.format(Locale.ROOT,
                "%s,%d,%d,%d,%d,%d,%.10g,%.10g,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                activeMeshProfile, activeBackgroundLevel, statistics.getNumElem(),
                statistics.getNumElem("tet"), statistics.getNumElem("pyr"),
                statistics.getNumElem("prism"), statistics.getMinQuality(),
                statistics.getMeanQuality(),
                local == null ? "" : local[0],
                local == null ? "" : local[1],
                local == null ? "" : local[2],
                local == null ? "" : local[3],
                local == null ? "" : local[4],
                local == null ? "" : local[5],
                local == null ? "" : local[6],
                local == null ? "" : local[7],
                local == null ? "" : local[8],
                local == null ? "" : local[9]);
        // COMSOL batch restricts direct java.nio writes. Emit a stable record;
        // the Python orchestration layer owns filesystem publication.
        System.out.println("MESH_CSV_LABEL," + outputLabel);
        System.out.println("MESH_CSV_HEADER," + header);
        System.out.println("MESH_CSV_ROW," + row);
    }


    private static void activateTransientInputs(Model model) {
        createZeroOrderHold(model, "p_cmd_nl", "power_cmd", SCHEDULE_TIME,
                SCHEDULE_POWER, "W");
        createZeroOrderHold(model, "ta_cmd_nl", "coolant_cmd", SCHEDULE_TIME,
                SCHEDULE_COOLANT_KELVIN, "K");
        createZeroOrderHold(model, "u_cmd_nl", "velocity_cmd", SCHEDULE_TIME,
                SCHEDULE_VELOCITY, "m/s");
        createZeroOrderHold(model, "ph_cmd_nl", "hidden_power_cmd", SCHEDULE_TIME,
                SCHEDULE_HIDDEN_POWER, "W");
        createZeroOrderHold(model, "ue_cmd_nl", "effective_velocity_cmd", SCHEDULE_TIME,
                SCHEDULE_EFFECTIVE_VELOCITY, "m/s");

        model.component("comp1").physics("ht").feature("hs1")
                .set("P0", "power_cmd(t)+hidden_power_cmd(t)");
        model.component("comp1").common("ampr1").set("T_amb", "coolant_cmd(t)");
        model.component("comp1").physics("spf").feature("inl1")
                .set("Uavfdf", "effective_velocity_cmd(t)");
        if (radiationEnabled) {
            model.component("comp1").physics("rad").feature("dsurf1")
                    .set("Tamb", "coolant_cmd(t)");
        }
    }

    private static void createZeroOrderHold(
            Model model, String tag, String functionName, List<String> time,
            List<String> values, String valueUnit) {
        String[][] pieces = new String[time.size() - 1][3];
        for (int i = 0; i < time.size() - 1; i++) {
            pieces[i][0] = time.get(i);
            pieces[i][1] = time.get(i + 1);
            pieces[i][2] = values.get(i);
        }
        model.func().create(tag, "Piecewise");
        model.func(tag).set("funcname", functionName);
        model.func(tag).set("arg", "t");
        model.func(tag).set("pieces", pieces);
        model.func(tag).set("argunit", "s");
        model.func(tag).set("fununit", valueUnit);
        model.func(tag).set("smooth", "none");
        model.func(tag).set("extrap", "constant");
    }

    private static void createAverage(
            Model model, String tag, String operatorName, int[] domains) {
        model.component("comp1").cpl().create(tag, "Average");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().geom("geom1", 3);
        model.component("comp1").cpl(tag).selection().set(domains);
    }

    private static void createAverageNamed(
            Model model, String tag, String operatorName, String selection) {
        model.component("comp1").cpl().create(tag, "Average");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().named(selection);
    }

    private static void createAverageBoundaryNamed(
            Model model, String tag, String operatorName, String selection) {
        model.component("comp1").cpl().create(tag, "Average");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().geom("geom1", 2);
        model.component("comp1").cpl(tag).selection().named(selection);
    }

    private static void createMaximum(
            Model model, String tag, String operatorName, int[] domains) {
        model.component("comp1").cpl().create(tag, "Maximum");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().geom("geom1", 3);
        model.component("comp1").cpl(tag).selection().set(domains);
    }

    private static void createMaximumNamed(
            Model model, String tag, String operatorName, String selection) {
        model.component("comp1").cpl().create(tag, "Maximum");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().named(selection);
    }

    private static void createIntegration(
            Model model, String tag, String operatorName, int[] domains) {
        model.component("comp1").cpl().create(tag, "Integration");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().geom("geom1", 3);
        model.component("comp1").cpl(tag).selection().set(domains);
    }

    private static void createIntegrationNamed(
            Model model, String tag, String operatorName, String selection) {
        model.component("comp1").cpl().create(tag, "Integration");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().named(selection);
    }

    private static void createIntegrationBoundaryNamed(
            Model model, String tag, String operatorName, String selection) {
        model.component("comp1").cpl().create(tag, "Integration");
        model.component("comp1").cpl(tag).set("opname", operatorName);
        model.component("comp1").cpl(tag).selection().geom("geom1", 2);
        model.component("comp1").cpl(tag).selection().named(selection);
    }

    private static String createStationaryStudy(Model model) {
        Set<String> existingSolvers = new HashSet<>(Arrays.asList(model.sol().tags()));
        model.study().create("std_nonlinear_init");
        model.study("std_nonlinear_init").label("Dataset stationary initial field");
        model.study("std_nonlinear_init").create("stat", "Stationary");
        configureStudyStep(model, "std_nonlinear_init", "stat");
        model.study("std_nonlinear_init").createAutoSequences("all");
        return newestNonemptySolverTag(model, existingSolvers);
    }

    private static String createTransientStudy(Model model, String initialSolutionTag) {
        Set<String> existingSolvers = new HashSet<>(Arrays.asList(model.sol().tags()));
        model.study().create("std_nonlinear");
        model.study("std_nonlinear").label("Dataset transient: nonlinear chip cooling");
        model.study("std_nonlinear").create("time", "Transient");
        model.study("std_nonlinear").feature("time")
                .set("tlist", String.join(" ", SCHEDULE_TIME));
        configureStudyStep(model, "std_nonlinear", "time");
        model.study("std_nonlinear").createAutoSequences("all");
        String solutionTag = newestNonemptySolverTag(model, existingSolvers);
        List<String> solverFeatures = Arrays.asList(model.sol(solutionTag).feature().tags());
        if (!solverFeatures.contains("v1")) {
            throw new IllegalStateException("Transient solver has no Variables feature");
        }
        model.sol(solutionTag).feature("v1").set("initmethod", "sol");
        model.sol(solutionTag).feature("v1").set("initsol", initialSolutionTag);
        model.sol(solutionTag).feature("v1").set("initsoluse", "current");
        if (solverFeatures.contains("t1")) {
            model.sol(solutionTag).feature("t1").set("tstepsbdf", "strict");
            for (String childTag
                    : model.sol(solutionTag).feature("t1").feature().tags()) {
                if ("Segregated".equals(model.sol(solutionTag).feature("t1")
                        .feature(childTag).getType())) {
                    model.sol(solutionTag).feature("t1").feature(childTag)
                            .set("maxsegiter", MAX_TRANSIENT_SEGREGATED_ITERATIONS);
                }
            }
        }
        return solutionTag;
    }

    private static String newestNonemptySolverTag(Model model, Set<String> existing) {
        String[] tags = model.sol().tags();
        for (int i = tags.length - 1; i >= 0; i--) {
            if (!existing.contains(tags[i])
                    && model.sol(tags[i]).feature().tags().length > 0) {
                return tags[i];
            }
        }
        throw new IllegalStateException("COMSOL did not create a populated solver sequence");
    }

    private static String solutionDatasetTag(
            Model model, String solutionTag, Set<String> existing) {
        String[] tags = model.result().dataset().tags();
        for (int i = tags.length - 1; i >= 0; i--) {
            if (!existing.contains(tags[i])
                    && solutionTag.equals(
                            model.result().dataset(tags[i]).getString("solution"))) {
                return tags[i];
            }
        }
        throw new IllegalStateException(
                "COMSOL did not create a result dataset for " + solutionTag);
    }

    private static void configureStudyStep(Model model, String study, String step) {
        model.study(study).feature(step).setSolveFor("/physics/ht", true);
        model.study(study).feature(step).setSolveFor("/physics/spf", true);
        model.study(study).feature(step).setSolveFor("/physics/rad", radiationEnabled);
        model.study(study).feature(step).setSolveFor("/multiphysics/nitf1", true);
        model.study(study).feature(step)
                .setSolveFor("/multiphysics/htrad1", radiationEnabled);
    }

    private static void exportResults(
            Model model, String datasetTag, String output, boolean transientResult) {
        String radiationExpression = radiationEnabled
                ? "comp1.int_rad_sink(comp1.rad.rflux)" : "0[W]";
        String powerExpression = transientResult
                ? "power_cmd(t)" : SCHEDULE_POWER.get(0) + "[W]";
        String coolantExpression = transientResult
                ? "coolant_cmd(t)" : SCHEDULE_COOLANT_KELVIN.get(0) + "[K]";
        String velocityExpression = transientResult
                ? "velocity_cmd(t)" : SCHEDULE_VELOCITY.get(0) + "[m/s]";
        String hiddenPowerExpression = transientResult
                ? "hidden_power_cmd(t)" : SCHEDULE_HIDDEN_POWER.get(0) + "[W]";
        String effectiveVelocityExpression = transientResult
                ? "effective_velocity_cmd(t)"
                : SCHEDULE_EFFECTIVE_VELOCITY.get(0) + "[m/s]";
        List<String> expressions = new ArrayList<>();
        List<String> units = new ArrayList<>();
        List<String> descriptions = new ArrayList<>();
        if (!transientResult) {
            expressions.add("0[s]");
            units.add("s");
            descriptions.add("Time");
        }
        expressions.addAll(Arrays.asList(
                "comp1.avg_chip(comp1.T)",
                "comp1.avg_base(comp1.T)",
                "comp1.avg_fins(comp1.T)",
                "comp1.max_chip(comp1.T)",
                "comp1.max_fins(comp1.T)",
                "comp1.avg_outlet(comp1.T)",
                "comp1.avg_inlet(comp1.p)-comp1.avg_outlet(comp1.p)",
                radiationExpression,
                "comp1.ht.heatBalance",
                powerExpression,
                coolantExpression,
                velocityExpression,
                hiddenPowerExpression,
                effectiveVelocityExpression,
                "comp1.int_chip(1)",
                "comp1.int_base(1)",
                "comp1.int_fins(1)"));
        units.addAll(Arrays.asList(
                "K", "K", "K", "K", "K", "K", "Pa", "W", "W",
                "W", "K", "m/s", "W", "m/s", "m^3", "m^3", "m^3"));
        descriptions.addAll(Arrays.asList(
                "Chip volume-average temperature",
                "Heat-sink base volume-average temperature",
                "Heat-sink fins volume-average temperature",
                "Maximum chip temperature",
                "Maximum fins temperature",
                "Outlet air mean temperature",
                "Mean inlet-to-outlet pressure drop",
                "Net radiative heat rate on heat-sink walls",
                "Heat-transfer energy balance residual",
                "Commanded chip power",
                "Commanded inlet air temperature",
                "Public inlet air velocity",
                "Unobserved disturbance power",
                "Effective inlet air velocity applied to COMSOL",
                "Chip volume",
                "Heat-sink base volume",
                "Heat-sink fins volume"));
        model.result().table().create("tbl_nonlinear", "Table");
        model.result().table("tbl_nonlinear").label("Nonlinear chip cooling dataset");
        model.result().numerical().create("gev_nonlinear", "EvalGlobal");
        model.result().numerical("gev_nonlinear").set("data", datasetTag);
        model.result().numerical("gev_nonlinear")
                .set("expr", expressions.toArray(new String[0]));
        model.result().numerical("gev_nonlinear")
                .set("unit", units.toArray(new String[0]));
        model.result().numerical("gev_nonlinear")
                .set("descr", descriptions.toArray(new String[0]));
        model.result().numerical("gev_nonlinear").set("table", "tbl_nonlinear");
        model.result().numerical("gev_nonlinear").setResult();
        model.result().table("tbl_nonlinear").save(output);
    }
}
