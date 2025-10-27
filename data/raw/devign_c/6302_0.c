static void s390_hot_add_cpu(const int64_t id, Error **errp)

{

    MachineState *machine = MACHINE(qdev_get_machine());



    s390x_new_cpu(machine->cpu_model, id, errp);

}
