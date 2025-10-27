static void x86_cpuid_set_tsc_freq(Object *obj, Visitor *v, void *opaque,

                                   const char *name, Error **errp)

{

    X86CPU *cpu = X86_CPU(obj);

    const int64_t min = 0;

    const int64_t max = INT_MAX;

    int64_t value;



    visit_type_int(v, &value, name, errp);

    if (error_is_set(errp)) {

        return;

    }

    if (value < min || value > max) {

        error_set(errp, QERR_PROPERTY_VALUE_OUT_OF_RANGE, "",

                  name ? name : "null", value, min, max);

        return;

    }



    cpu->env.tsc_khz = value / 1000;

}
