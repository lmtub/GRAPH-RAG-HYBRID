static void pc_machine_set_vmport(Object *obj, bool value, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(obj);



    pcms->vmport = value;

}
