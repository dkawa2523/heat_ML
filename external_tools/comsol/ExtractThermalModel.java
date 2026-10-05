import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

/** Read a stored transient solution into heat capacities and representative temperatures. */
public final class ExtractThermalModel {
    private ExtractThermalModel() {}

    private static String decode(String text) {
        return new String(Base64.getDecoder().decode(text), StandardCharsets.UTF_8);
    }

    private static String dataset(Model model, String requested) {
        List<String> available = new ArrayList<>();
        for (String tag : model.result().dataset().tags()) {
            if ("Solution".equals(model.result().dataset(tag).getType())
                    && "Time".equals(model.sol(model.result().dataset(tag).getString("solution")).getType())) {
                available.add(tag);
            }
        }
        if (!requested.isEmpty() && available.contains(requested)) {
            return requested;
        }
        if (requested.isEmpty() && available.size() == 1) {
            return available.get(0);
        }
        throw new IllegalArgumentException(
                "Select a stored transient Solution dataset with --dataset; available: " + available);
    }

    private static String feature(Model model, String type, String dataset) {
        String tag = model.result().numerical().uniquetag("ct_import");
        model.result().numerical().create(tag, type);
        model.result().numerical(tag).set("data", dataset);
        model.result().numerical(tag).set("innerinput", "all");
        return tag;
    }

    private static double[] global(Model model, String dataset, String expression, String unit) {
        String tag = feature(model, "EvalGlobal", dataset);
        model.result().numerical(tag).set("expr", new String[]{expression});
        if (!unit.isEmpty()) {
            model.result().numerical(tag).set("unit", new String[]{unit});
        }
        double[] values = model.result().numerical(tag).getReal()[0];
        model.result().numerical().remove(tag);
        return values;
    }

    private static int[] domains(Model model, String component, String selection) {
        if (selection.matches("[0-9]+(,[0-9]+)*")) {
            return Arrays.stream(selection.split(",")).mapToInt(Integer::parseInt).toArray();
        }
        return model.component(component).selection(selection).entities(3);
    }

    private static void region(
            Model model, String dataset, String component, String capacity,
            String name, String selection, List<String> headers, List<double[]> columns) {
        int[] domains = domains(model, component, selection);
        if (domains.length == 0) {
            throw new IllegalArgumentException("Empty COMSOL domain selection for node " + name);
        }
        String[] geometries = model.component(component).geom().tags();
        if (geometries.length != 1) {
            throw new IllegalArgumentException("Select a component with one volume geometry");
        }
        String tag = feature(model, "IntVolume", dataset);
        model.result().numerical(tag).model(component);
        model.result().numerical(tag).selection().geom(geometries[0], 3);
        model.result().numerical(tag).selection().set(domains);
        model.result().numerical(tag).set("expr", new String[]{capacity, "(" + capacity + ")*T"});
        model.result().numerical(tag).set("unit", new String[]{"J/K", "J"});
        double[][] values = model.result().numerical(tag).getReal();
        headers.add("node." + name + ".heat_capacity");
        columns.add(values[0]);
        headers.add("node." + name + ".temperature_integral");
        columns.add(values[1]);
        model.result().numerical().remove(tag);
    }

    private static void write(Model model, String output, List<String> headers, List<double[]> columns) {
        int count = columns.get(0).length;
        for (double[] values : columns) {
            if (values.length != count) {
                throw new IllegalArgumentException("Stored result columns have different time grids");
            }
        }
        double[][] rows = new double[count][columns.size()];
        for (int row = 0; row < count; row++) {
            for (int col = 0; col < columns.size(); col++) {
                rows[row][col] = columns.get(col)[row];
            }
        }
        String tag = model.result().table().uniquetag("ct_import");
        model.result().table().create(tag, "Table");
        model.result().table(tag).setTableData(rows);
        model.result().table(tag).setColumnHeaders(headers.toArray(new String[0]));
        model.result().table(tag).save(output);
        model.result().table().remove(tag);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 6) {
            throw new IllegalArgumentException("Expected model, output and four encoded arguments");
        }
        Model model = ModelUtil.loadCopy("CelltempImport", args[0]);
        try {
            String component = decode(args[2]);
            String dataset = dataset(model, decode(args[3]));
            String solution = model.result().dataset(dataset).getString("solution");
            model.result().dataset(dataset).set("comp", component);
            if (model.sol(solution).getSolutioninfo().getOuterSolnum().length > 1) {
                throw new IllegalArgumentException("Choose a solution without an outer parameter sweep");
            }
            List<String> headers = new ArrayList<>();
            List<double[]> columns = new ArrayList<>();
            headers.add("time");
            columns.add(global(model, dataset, "t", "s"));
            for (String row : decode(args[5]).split("\n")) {
                String[] fields = row.split("\t", -1);
                String name = decode(fields[1]);
                String value = decode(fields[2]);
                if ("region".equals(fields[0])) {
                    region(model, dataset, component, decode(args[4]), name, value, headers, columns);
                } else {
                    headers.add("control." + name + ".command");
                    columns.add(global(model, dataset, value, decode(fields[3])));
                }
            }
            write(model, args[1], headers, columns);
        } finally {
            ModelUtil.remove("CelltempImport");
        }
    }
}
