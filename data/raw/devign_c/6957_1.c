static void trigger_console_data(void *opaque, int n, int level)

{

    sclp_service_interrupt(0);

}
