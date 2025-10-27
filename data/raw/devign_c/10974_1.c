static void qxl_spice_monitors_config_async(PCIQXLDevice *qxl, int replay)

{

    trace_qxl_spice_monitors_config(qxl->id);

    if (replay) {

        /*

         * don't use QXL_COOKIE_TYPE_IO:

         *  - we are not running yet (post_load), we will assert

         *    in send_events

         *  - this is not a guest io, but a reply, so async_io isn't set.

         */

        spice_qxl_monitors_config_async(&qxl->ssd.qxl,

                qxl->guest_monitors_config,

                MEMSLOT_GROUP_GUEST,

                (uintptr_t)qxl_cookie_new(

                    QXL_COOKIE_TYPE_POST_LOAD_MONITORS_CONFIG,

                    0));

    } else {

#if SPICE_SERVER_VERSION >= 0x000c06 /* release 0.12.6 */

        if (qxl->max_outputs) {

            spice_qxl_set_monitors_config_limit(&qxl->ssd.qxl,

                                                qxl->max_outputs);

        }

#endif

        qxl->guest_monitors_config = qxl->ram->monitors_config;

        spice_qxl_monitors_config_async(&qxl->ssd.qxl,

                qxl->ram->monitors_config,

                MEMSLOT_GROUP_GUEST,

                (uintptr_t)qxl_cookie_new(QXL_COOKIE_TYPE_IO,

                                          QXL_IO_MONITORS_CONFIG_ASYNC));

    }

}
