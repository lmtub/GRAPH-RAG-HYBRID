AlphaCPU *cpu_alpha_init(const char *cpu_model)

{

    AlphaCPU *cpu;

    ObjectClass *cpu_class;



    cpu_class = alpha_cpu_class_by_name(cpu_model);

    if (cpu_class == NULL) {

        /* Default to ev67; no reason not to emulate insns by default.  */

        cpu_class = object_class_by_name(TYPE("ev67"));

    }

    cpu = ALPHA_CPU(object_new(object_class_get_name(cpu_class)));



    object_property_set_bool(OBJECT(cpu), true, "realized", NULL);



    return cpu;

}
