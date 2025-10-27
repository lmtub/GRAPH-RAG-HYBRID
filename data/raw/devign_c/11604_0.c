static void pc_machine_set_nvdimm(Object *obj, bool value, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(obj);



    pcms->nvdimm = value;

}
