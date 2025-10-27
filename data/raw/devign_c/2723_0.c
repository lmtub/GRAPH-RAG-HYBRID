void net_host_device_add(Monitor *mon, const char *device, const char *opts)

{

    if (!net_host_check_device(device)) {

        monitor_printf(mon, "invalid host network device %s\n", device);

        return;

    }

    if (net_client_init(device, opts ? : "") < 0) {

        monitor_printf(mon, "adding host network device %s failed\n", device);

    }

}
