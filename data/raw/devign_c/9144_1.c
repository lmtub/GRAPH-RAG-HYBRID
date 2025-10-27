static void panicked_mon_event(const char *action)

{

    QObject *data;



    data = qobject_from_jsonf("{ 'action': %s }", action);

    monitor_protocol_event(QEVENT_GUEST_PANICKED, data);

    qobject_decref(data);

}
