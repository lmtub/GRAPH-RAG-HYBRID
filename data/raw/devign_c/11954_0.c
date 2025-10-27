static bool pc_machine_get_vmport(Object *obj, Error **errp)

{

    PCMachineState *pcms = PC_MACHINE(obj);



    return pcms->vmport;

}
