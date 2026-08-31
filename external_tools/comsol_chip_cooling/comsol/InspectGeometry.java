import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/** Print volume moments used to verify final domain roles. */
public final class InspectGeometry {
    private InspectGeometry() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: InspectGeometry <source.mph>");
        }
        Model model = ModelUtil.loadCopy("GeometryInspection", args[0]);
        for (int domain = 2; domain <= 7; domain++) {
            String tag = "volume_d" + domain;
            model.result().numerical().create(tag, "IntVolume");
            model.result().numerical(tag).set("data", "dset2");
            model.result().numerical(tag).selection().set(domain);
            model.result().numerical(tag).set("expr", new String[]{"1", "x", "y", "z"});
            double[][] values = model.result().numerical(tag).getReal();
            System.out.println("Domain " + domain + " integrals [V, int(x), int(y), int(z)]: "
                    + java.util.Arrays.deepToString(values));
        }
        System.out.println("Final named Chip domains: "
                + java.util.Arrays.toString(model.component("comp1").selection("sel1").entities(3)));
        ModelUtil.remove("GeometryInspection");
    }
}
