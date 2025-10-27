static void pc_cpu_unplug_request_cb(HotplugHandler *hotplug_dev,

                                     DeviceState *dev, Error **errp)

{

    int idx = -1;

    HotplugHandlerClass *hhc;

    Error *local_err = NULL;

    PCMachineState *pcms = PC_MACHINE(hotplug_dev);



    pc_find_cpu_slot(pcms, CPU(dev), &idx);

    assert(idx != -1);

    if (idx == 0) {

        error_setg(&local_err, "Boot CPU is unpluggable");

        goto out;

    }



    if (idx < pcms->possible_cpus->len - 1 &&

        pcms->possible_cpus->cpus[idx + 1].cpu != NULL) {

        X86CPU *cpu;



        for (idx = pcms->possible_cpus->len - 1;

             pcms->possible_cpus->cpus[idx].cpu == NULL; idx--) {

            ;;

        }



        cpu = X86_CPU(pcms->possible_cpus->cpus[idx].cpu);

        error_setg(&local_err, "CPU [socket-id: %u, core-id: %u,"

                   " thread-id: %u] should be removed first",

                   cpu->socket_id, cpu->core_id, cpu->thread_id);

        goto out;



    }



    hhc = HOTPLUG_HANDLER_GET_CLASS(pcms->acpi_dev);

    hhc->unplug_request(HOTPLUG_HANDLER(pcms->acpi_dev), dev, &local_err);



    if (local_err) {

        goto out;

    }



 out:

    error_propagate(errp, local_err);



}
