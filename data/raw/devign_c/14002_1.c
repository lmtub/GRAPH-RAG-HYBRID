static void qmp_monitor_complete(void *opaque, QObject *ret_data)

{

    monitor_protocol_emitter(opaque, ret_data);

}
