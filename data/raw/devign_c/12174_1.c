static void pnv_chip_realize(DeviceState *dev, Error **errp)
{
    PnvChip *chip = PNV_CHIP(dev);
    Error *error = NULL;
    PnvChipClass *pcc = PNV_CHIP_GET_CLASS(chip);
    char *typename = pnv_core_typename(pcc->cpu_model);
    size_t typesize = object_type_get_instance_size(typename);
    int i, core_hwid;
    if (!object_class_by_name(typename)) {
        error_setg(errp, "Unable to find PowerNV CPU Core '%s'", typename);
    /* Cores */
    pnv_chip_core_sanitize(chip, &error);
    chip->cores = g_malloc0(typesize * chip->nr_cores);
    for (i = 0, core_hwid = 0; (core_hwid < sizeof(chip->cores_mask) * 8)
             && (i < chip->nr_cores); core_hwid++) {
        char core_name[32];
        void *pnv_core = chip->cores + i * typesize;
        if (!(chip->cores_mask & (1ull << core_hwid))) {
            continue;
        object_initialize(pnv_core, typesize, typename);
        snprintf(core_name, sizeof(core_name), "core[%d]", core_hwid);
        object_property_add_child(OBJECT(chip), core_name, OBJECT(pnv_core),
                                  &error_fatal);
        object_property_set_int(OBJECT(pnv_core), smp_threads, "nr-threads",
                                &error_fatal);
        object_property_set_int(OBJECT(pnv_core), core_hwid,
                                CPU_CORE_PROP_CORE_ID, &error_fatal);
        object_property_set_int(OBJECT(pnv_core),
                                pcc->core_pir(chip, core_hwid),
                                "pir", &error_fatal);
        object_property_set_bool(OBJECT(pnv_core), true, "realized",
                                 &error_fatal);
        object_unref(OBJECT(pnv_core));
        i++;
    g_free(typename);