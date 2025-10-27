static void os_host_main_loop_wait(int *timeout)

{

    int ret, ret2, i;

    PollingEntry *pe;



    /* XXX: need to suppress polling by better using win32 events */

    ret = 0;

    for (pe = first_polling_entry; pe != NULL; pe = pe->next) {

        ret |= pe->func(pe->opaque);

    }

    if (ret == 0) {

        int err;

        WaitObjects *w = &wait_objects;



        qemu_mutex_unlock_iothread();

        ret = WaitForMultipleObjects(w->num, w->events, FALSE, *timeout);

        qemu_mutex_lock_iothread();

        if (WAIT_OBJECT_0 + 0 <= ret && ret <= WAIT_OBJECT_0 + w->num - 1) {

            if (w->func[ret - WAIT_OBJECT_0]) {

                w->func[ret - WAIT_OBJECT_0](w->opaque[ret - WAIT_OBJECT_0]);

            }



            /* Check for additional signaled events */

            for (i = (ret - WAIT_OBJECT_0 + 1); i < w->num; i++) {

                /* Check if event is signaled */

                ret2 = WaitForSingleObject(w->events[i], 0);

                if (ret2 == WAIT_OBJECT_0) {

                    if (w->func[i]) {

                        w->func[i](w->opaque[i]);

                    }

                } else if (ret2 != WAIT_TIMEOUT) {

                    err = GetLastError();

                    fprintf(stderr, "WaitForSingleObject error %d %d\n", i, err);

                }

            }

        } else if (ret != WAIT_TIMEOUT) {

            err = GetLastError();

            fprintf(stderr, "WaitForMultipleObjects error %d %d\n", ret, err);

        }

    }



    *timeout = 0;

}
