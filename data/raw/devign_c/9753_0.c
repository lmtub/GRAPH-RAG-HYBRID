static void main_loop(void)

{

    int r;



#ifdef CONFIG_IOTHREAD

    qemu_system_ready = 1;

    qemu_cond_broadcast(&qemu_system_cond);

#endif



    for (;;) {

        do {

#ifdef CONFIG_PROFILER

            int64_t ti;

#endif

#ifndef CONFIG_IOTHREAD

            tcg_cpu_exec();

#endif

#ifdef CONFIG_PROFILER

            ti = profile_getclock();

#endif

            main_loop_wait(qemu_calculate_timeout());

#ifdef CONFIG_PROFILER

            dev_time += profile_getclock() - ti;

#endif

        } while (vm_can_run());



        if (qemu_debug_requested()) {

            monitor_protocol_event(QEVENT_DEBUG, NULL);

            vm_stop(EXCP_DEBUG);

        }

        if (qemu_shutdown_requested()) {

            monitor_protocol_event(QEVENT_SHUTDOWN, NULL);

            if (no_shutdown) {

                vm_stop(0);

                no_shutdown = 0;

            } else

                break;

        }

        if (qemu_reset_requested()) {

            monitor_protocol_event(QEVENT_RESET, NULL);

            pause_all_vcpus();

            qemu_system_reset();

            resume_all_vcpus();

        }

        if (qemu_powerdown_requested()) {

            monitor_protocol_event(QEVENT_POWERDOWN, NULL);

            qemu_irq_raise(qemu_system_powerdown);

        }

        if ((r = qemu_vmstop_requested())) {

            vm_stop(r);

        }

    }

    pause_all_vcpus();

}
