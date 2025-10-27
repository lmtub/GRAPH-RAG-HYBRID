int vm_stop(RunState state)

{

    if (qemu_in_vcpu_thread()) {


        qemu_system_vmstop_request(state);

        /*

         * FIXME: should not return to device code in case

         * vm_stop() has been requested.

         */

        cpu_stop_current();

        return 0;

    }



    return do_vm_stop(state);

}