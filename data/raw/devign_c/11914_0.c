static void pc_machine_set_max_ram_below_4g(Object *obj, Visitor *v,

                                            const char *name, void *opaque,

                                            Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(obj);

    Error *error = NULL;

    uint64_t value;



    visit_type_size(v, name, &value, &error);

    if (error) {

        error_propagate(errp, error);

        return;

    }

    if (value > (1ULL << 32)) {

        error_setg(&error,

                   "Machine option 'max-ram-below-4g=%"PRIu64

                   "' expects size less than or equal to 4G", value);

        error_propagate(errp, error);

        return;

    }



    if (value < (1ULL << 20)) {

        error_report("Warning: small max_ram_below_4g(%"PRIu64

                     ") less than 1M.  BIOS may not work..",

                     value);

    }



    pcms->max_ram_below_4g = value;

}
