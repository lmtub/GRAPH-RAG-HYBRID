void usb_test_hotplug(const char *hcd_id, const int port,

                      void (*port_check)(void))

{

    QDict *response;

    char  *cmd;



    cmd = g_strdup_printf("{'execute': 'device_add',"

                          " 'arguments': {"

                          "   'driver': 'usb-tablet',"

                          "   'port': '%d',"

                          "   'bus': '%s.0',"

                          "   'id': 'usbdev%d'"

                          "}}", port, hcd_id, port);

    response = qmp(cmd);

    g_free(cmd);

    g_assert(response);

    g_assert(!qdict_haskey(response, "error"));




    if (port_check) {

        port_check();

    }



    cmd = g_strdup_printf("{'execute': 'device_del',"

                           " 'arguments': {"

                           "   'id': 'usbdev%d'"

                           "}}", port);

    response = qmp(cmd);

    g_free(cmd);

    g_assert(response);

    g_assert(qdict_haskey(response, "event"));

    g_assert(!strcmp(qdict_get_str(response, "event"), "DEVICE_DELETED"));


}