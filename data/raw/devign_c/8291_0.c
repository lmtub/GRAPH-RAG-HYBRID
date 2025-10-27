S390CPU *s390x_new_cpu(const char *cpu_model, uint32_t core_id, Error **errp)

{

    S390CPU *cpu;

    Error *err = NULL;



    cpu = cpu_s390x_create(cpu_model, &err);

    if (err != NULL) {

        goto out;

    }



    object_property_set_int(OBJECT(cpu), core_id, "core-id", &err);

    if (err != NULL) {

        goto out;

    }

    object_property_set_bool(OBJECT(cpu), true, "realized", &err);



out:

    if (err) {

        error_propagate(errp, err);

        object_unref(OBJECT(cpu));

        cpu = NULL;

    }

    return cpu;

}
