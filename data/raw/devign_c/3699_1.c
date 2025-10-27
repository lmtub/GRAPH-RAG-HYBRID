static void acpi_get_pm_info(AcpiPmInfo *pm)

{

    Object *piix = piix4_pm_find();

    Object *lpc = ich9_lpc_find();

    Object *obj = NULL;

    QObject *o;




    pm->pcihp_io_base = 0;

    pm->pcihp_io_len = 0;

    if (piix) {

        obj = piix;

        pm->cpu_hp_io_base = PIIX4_CPU_HOTPLUG_IO_BASE;

        pm->pcihp_io_base =

            object_property_get_int(obj, ACPI_PCIHP_IO_BASE_PROP, NULL);

        pm->pcihp_io_len =

            object_property_get_int(obj, ACPI_PCIHP_IO_LEN_PROP, NULL);

    }

    if (lpc) {

        obj = lpc;

        pm->cpu_hp_io_base = ICH9_CPU_HOTPLUG_IO_BASE;

    }

    assert(obj);



    pm->cpu_hp_io_len = ACPI_GPE_PROC_LEN;

    pm->mem_hp_io_base = ACPI_MEMORY_HOTPLUG_BASE;

    pm->mem_hp_io_len = ACPI_MEMORY_HOTPLUG_IO_LEN;



    /* Fill in optional s3/s4 related properties */

    o = object_property_get_qobject(obj, ACPI_PM_PROP_S3_DISABLED, NULL);

    if (o) {

        pm->s3_disabled = qint_get_int(qobject_to_qint(o));

    } else {

        pm->s3_disabled = false;

    }

    qobject_decref(o);

    o = object_property_get_qobject(obj, ACPI_PM_PROP_S4_DISABLED, NULL);

    if (o) {

        pm->s4_disabled = qint_get_int(qobject_to_qint(o));

    } else {

        pm->s4_disabled = false;

    }

    qobject_decref(o);

    o = object_property_get_qobject(obj, ACPI_PM_PROP_S4_VAL, NULL);

    if (o) {

        pm->s4_val = qint_get_int(qobject_to_qint(o));

    } else {

        pm->s4_val = false;

    }

    qobject_decref(o);



    /* Fill in mandatory properties */

    pm->sci_int = object_property_get_int(obj, ACPI_PM_PROP_SCI_INT, NULL);



    pm->acpi_enable_cmd = object_property_get_int(obj,

                                                  ACPI_PM_PROP_ACPI_ENABLE_CMD,

                                                  NULL);

    pm->acpi_disable_cmd = object_property_get_int(obj,

                                                  ACPI_PM_PROP_ACPI_DISABLE_CMD,

                                                  NULL);

    pm->io_base = object_property_get_int(obj, ACPI_PM_PROP_PM_IO_BASE,

                                          NULL);

    pm->gpe0_blk = object_property_get_int(obj, ACPI_PM_PROP_GPE0_BLK,

                                           NULL);

    pm->gpe0_blk_len = object_property_get_int(obj, ACPI_PM_PROP_GPE0_BLK_LEN,

                                               NULL);

    pm->pcihp_bridge_en =

        object_property_get_bool(obj, "acpi-pci-hotplug-with-bridge-support",

                                 NULL);

}