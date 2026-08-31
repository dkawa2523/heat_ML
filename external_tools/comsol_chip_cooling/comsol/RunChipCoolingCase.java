import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Create and solve one transient chip-cooling case from the COMSOL Application
 * Library model.  The installed source model is opened as a copy and is never
 * overwritten.
 *
 * <p>The schedule is passed as base64-encoded CSV because COMSOL's default
 * Java sandbox intentionally blocks direct arbitrary-file reads. CSV columns are:
 * time_s, chip_power_w, coolant_temperature_c, hidden_power_w,
 * initial_temperature_c. The last two columns are optional. Initial
 * temperature is read from the first data row and otherwise defaults to the
 * first coolant temperature.</p>
 */
public final class RunChipCoolingCase {
    private static final int[] SOLID_DOMAINS = {2, 3, 4, 5, 6, 7};
    private static final int BASE_DOMAIN = 2;
    private static final int[] FIN_DOMAINS = {3, 5, 6, 7};

    private RunChipCoolingCase() {}

    // Keep the runner as one compiled class.  comsolcompile does not emit
    // companion class files for nested utility classes in this execution mode.
    private static final List<String> SCHEDULE_TIME = new ArrayList<>();
    private static final List<String> SCHEDULE_POWER = new ArrayList<>();
    private static final List<String> SCHEDULE_COOLANT_KELVIN = new ArrayList<>();
    private static final List<String> SCHEDULE_HIDDEN_POWER = new ArrayList<>();
    private static String initialTemperatureKelvin;

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            throw new IllegalArgumentException(
                    "Usage: RunChipCoolingCase <source.mph> <schedule-base64> "
                            + "<raw-output.txt> <edited-model.mph|->");
        }

        String sourceModel = args[0];
        String encodedSchedule = args[1];
        String rawOutput = args[2];
        String editedModelOutput = args[3];
        readSchedule(encodedSchedule);

        Model model = ModelUtil.loadCopy("ChipCoolingDataset", sourceModel);
        Set<String> existingDatasets = new HashSet<>(Arrays.asList(
                model.result().dataset().tags()));
        configureModel(model);
        String solutionTag = createTransientStudy(model);

        System.out.println("Solving transient study with " + SCHEDULE_TIME.size()
                + " requested output times (" + solutionTag + ")");
        model.sol(solutionTag).runAll();

        String datasetTag = newestTag(
                model.result().dataset().tags(), existingDatasets,
                "result dataset");
        exportResults(model, datasetTag, rawOutput);

        if (!"-".equals(editedModelOutput)) {
            model.save(editedModelOutput);
            System.out.println("Saved edited, solved model to " + editedModelOutput);
        }
        System.out.println("Saved raw result table to " + rawOutput);
        ModelUtil.remove("ChipCoolingDataset");
    }

    private static void readSchedule(String encoded) {
        String csv = new String(Base64.getDecoder().decode(encoded), StandardCharsets.UTF_8);
        String[] lines = csv.replace("\r", "").split("\n");
        if (lines.length < 2) {
            throw new IllegalArgumentException("Schedule is empty");
        }
        double previousTime = Double.NEGATIVE_INFINITY;
        for (int i = 1; i < lines.length; i++) {
            if (lines[i].trim().isEmpty()) {
                continue;
            }
            int lineNumber = i + 1;
            String[] fields = lines[i].split(",", -1);
            if (fields.length < 3) {
                throw new IllegalArgumentException(
                        "Expected at least 3 columns at line " + lineNumber);
            }
            double time = Double.parseDouble(fields[0].trim());
            double power = Double.parseDouble(fields[1].trim());
            double coolantCelsius = Double.parseDouble(fields[2].trim());
            double hidden = fields.length >= 4 && !fields[3].trim().isEmpty()
                    ? Double.parseDouble(fields[3].trim()) : 0.0;
            double initialCelsius = fields.length >= 5 && !fields[4].trim().isEmpty()
                    ? Double.parseDouble(fields[4].trim()) : coolantCelsius;
            if (!(time > previousTime)) {
                throw new IllegalArgumentException(
                        "Schedule time must be strictly increasing at line " + lineNumber);
            }
            if (power < 0.0 || hidden < 0.0) {
                throw new IllegalArgumentException(
                        "Power must be non-negative at line " + lineNumber);
            }
            SCHEDULE_TIME.add(Double.toString(time));
            SCHEDULE_POWER.add(Double.toString(power));
            SCHEDULE_COOLANT_KELVIN.add(Double.toString(coolantCelsius + 273.15));
            SCHEDULE_HIDDEN_POWER.add(Double.toString(hidden));
            if (initialTemperatureKelvin == null) {
                initialTemperatureKelvin = Double.toString(initialCelsius + 273.15);
            }
            previousTime = time;
        }
        if (SCHEDULE_TIME.size() < 2 || Double.parseDouble(SCHEDULE_TIME.get(0)) != 0.0) {
            throw new IllegalArgumentException(
                    "Schedule must contain at least two rows and begin at time 0");
        }
    }

    private static void configureModel(Model model) {
        System.out.println("Final named Chip domains: " + Arrays.toString(
                model.component("comp1").selection("sel1").entities(3)));
        // The reduced-model reference dataset deliberately isolates conduction,
        // interface resistance, thermal storage, and external convection.  The
        // stock air-flow and radiation interfaces remain in the copy but are not
        // solved by the transient dataset study.
        model.component("comp1").physics("ht").selection().set(SOLID_DOMAINS);
        model.component("comp1").physics("ht").feature("hs1")
                .set("P0", "power_cmd(t)+hidden_power_cmd(t)");
        model.component("comp1").physics("ht").feature("hf1")
                .set("Text_src", "userdef");
        model.component("comp1").physics("ht").feature("hf1")
                .set("Text", "coolant_cmd(t)");
        model.component("comp1").physics("ht").feature("init1")
                .set("Tinit", initialTemperatureKelvin + "[K]");

        // Replace the tutorial's silica chip with a constant-property silicon
        // surrogate.  Constant properties preserve the linearity expected by
        // the current RC estimator while keeping semiconductor-relevant scale.
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

        createZeroOrderHold(model, "p_cmd", "power_cmd", SCHEDULE_TIME, SCHEDULE_POWER, "W");
        createZeroOrderHold(model, "ta_cmd", "coolant_cmd", SCHEDULE_TIME,
                SCHEDULE_COOLANT_KELVIN, "K");
        createZeroOrderHold(model, "ph_cmd", "hidden_power_cmd", SCHEDULE_TIME,
                SCHEDULE_HIDDEN_POWER, "W");

        createAverageNamed(model, "avg_chip_cpl", "avg_chip", "sel1");
        createAverage(model, "avg_base_cpl", "avg_base", new int[]{BASE_DOMAIN});
        createAverage(model, "avg_fins_cpl", "avg_fins", FIN_DOMAINS);
        createMaximumNamed(model, "max_chip_cpl", "max_chip", "sel1");
        createIntegrationNamed(model, "int_chip_cpl", "int_chip", "sel1");
        createIntegration(model, "int_base_cpl", "int_base", new int[]{BASE_DOMAIN});
        createIntegration(model, "int_fins_cpl", "int_fins", FIN_DOMAINS);
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

    private static String createTransientStudy(Model model) {
        Set<String> existingSolvers = new HashSet<>(Arrays.asList(model.sol().tags()));
        model.study().create("std_data");
        model.study("std_data").label("Dataset transient: solid cooling");
        model.study("std_data").create("time", "Transient");
        model.study("std_data").feature("time").set("tlist", String.join(" ", SCHEDULE_TIME));
        model.study("std_data").feature("time").setSolveFor("/physics/ht", true);
        model.study("std_data").feature("time").setSolveFor("/physics/spf", false);
        model.study("std_data").feature("time").setSolveFor("/physics/rad", false);
        model.study("std_data").feature("time").setSolveFor("/multiphysics/nitf1", false);
        model.study("std_data").feature("time").setSolveFor("/multiphysics/htrad1", false);
        model.study("std_data").createAutoSequences("all");
        String solutionTag = newestTag(model.sol().tags(), existingSolvers, "solver sequence");
        // Controls may change at requested output times. Strict BDF stepping
        // prevents a large free step from crossing a command corner and then
        // back-interpolating a thermally noncausal output value.
        model.sol(solutionTag).feature("t1").set("tstepsbdf", "strict");
        return solutionTag;
    }

    private static String newestTag(String[] tags, Set<String> existing, String kind) {
        for (int i = tags.length - 1; i >= 0; i--) {
            if (!existing.contains(tags[i])) {
                return tags[i];
            }
        }
        throw new IllegalStateException("COMSOL did not create a new " + kind);
    }

    private static void exportResults(Model model, String datasetTag, String output) {
        model.result().table().create("tbl_data", "Table");
        model.result().table("tbl_data").label("Chip cooling dataset");
        model.result().numerical().create("gev_data", "EvalGlobal");
        model.result().numerical("gev_data").set("data", datasetTag);
        model.result().numerical("gev_data").set("expr", new String[]{
                "comp1.avg_chip(comp1.T)",
                "comp1.avg_base(comp1.T)",
                "comp1.avg_fins(comp1.T)",
                "comp1.max_chip(comp1.T)",
                "power_cmd(t)",
                "coolant_cmd(t)",
                "hidden_power_cmd(t)",
                "comp1.int_chip(1)",
                "comp1.int_base(1)",
                "comp1.int_fins(1)"});
        model.result().numerical("gev_data").set("unit", new String[]{
                "K", "K", "K", "K", "W", "K", "W", "m^3", "m^3", "m^3"});
        model.result().numerical("gev_data").set("descr", new String[]{
                "Chip volume-average temperature",
                "Heat-sink base volume-average temperature",
                "Heat-sink fins volume-average temperature",
                "Maximum chip temperature",
                "Commanded chip power",
                "Commanded coolant temperature",
                "Unobserved disturbance power",
                "Chip volume",
                "Heat-sink base volume",
                "Heat-sink fins volume"});
        model.result().numerical("gev_data").set("table", "tbl_data");
        model.result().numerical("gev_data").setResult();
        model.result().table("tbl_data").save(output);
    }
}
