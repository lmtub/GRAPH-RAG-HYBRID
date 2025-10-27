void qemu_system_reset(bool report)

{

    MachineClass *mc;



    mc = current_machine ? MACHINE_GET_CLASS(current_machine) : NULL;



    cpu_synchronize_all_states();



    if (mc && mc->reset) {

        mc->reset();

    } else {

        qemu_devices_reset();

    }

    if (report) {

        qapi_event_send_reset(&error_abort);

    }

    cpu_synchronize_all_post_reset();

}
