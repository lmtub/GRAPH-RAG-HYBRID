static void spapr_machine_2_3_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);



    mc->name = "pseries-2.3";

    mc->desc = "pSeries Logical Partition (PAPR compliant) v2.3";

    mc->alias = "pseries";

    mc->is_default = 1;

}
