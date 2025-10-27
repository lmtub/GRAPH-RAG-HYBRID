static void qemu_event_increment(void)

{

    SetEvent(qemu_event_handle);

}
