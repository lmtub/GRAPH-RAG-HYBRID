CPUState *cpu_generic_init(const char *typename, const char *cpu_model)

{

    /* TODO: all callers of cpu_generic_init() need to be converted to

     * call cpu_parse_features() only once, before calling cpu_generic_init().

     */

    const char *cpu_type = cpu_parse_cpu_model(typename, cpu_model);



    if (cpu_type) {

        return cpu_create(cpu_type);

    }

    return NULL;

}
