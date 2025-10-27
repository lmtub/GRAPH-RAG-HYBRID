static void cpu_x86_register(X86CPU *cpu, const char *name, Error **errp)

{

    CPUX86State *env = &cpu->env;

    x86_def_t def1, *def = &def1;



    memset(def, 0, sizeof(*def));



    if (cpu_x86_find_by_name(cpu, def, name) < 0) {

        error_setg(errp, "Unable to find CPU definition: %s", name);

        return;

    }



    object_property_set_str(OBJECT(cpu), def->vendor, "vendor", errp);

    object_property_set_int(OBJECT(cpu), def->level, "level", errp);

    object_property_set_int(OBJECT(cpu), def->family, "family", errp);

    object_property_set_int(OBJECT(cpu), def->model, "model", errp);

    object_property_set_int(OBJECT(cpu), def->stepping, "stepping", errp);

    env->features[FEAT_1_EDX] = def->features[FEAT_1_EDX];

    env->features[FEAT_1_ECX] = def->features[FEAT_1_ECX];

    env->features[FEAT_8000_0001_EDX] = def->features[FEAT_8000_0001_EDX];

    env->features[FEAT_8000_0001_ECX] = def->features[FEAT_8000_0001_ECX];

    object_property_set_int(OBJECT(cpu), def->xlevel, "xlevel", errp);

    env->features[FEAT_KVM] = def->features[FEAT_KVM];

    env->features[FEAT_SVM] = def->features[FEAT_SVM];

    env->features[FEAT_C000_0001_EDX] = def->features[FEAT_C000_0001_EDX];

    env->features[FEAT_7_0_EBX] = def->features[FEAT_7_0_EBX];

    env->cpuid_xlevel2 = def->xlevel2;

    cpu->cache_info_passthrough = def->cache_info_passthrough;



    object_property_set_str(OBJECT(cpu), def->model_id, "model-id", errp);



    /* Special cases not set in the x86_def_t structs: */

    if (kvm_enabled()) {

        env->features[FEAT_KVM] |= kvm_default_features;

    }

    env->features[FEAT_1_ECX] |= CPUID_EXT_HYPERVISOR;

}
