int net_handle_fd_param(Monitor *mon, const char *param)

{

    if (!qemu_isdigit(param[0])) {

        int fd;



        fd = monitor_get_fd(mon, param);

        if (fd == -1) {

            error_report("No file descriptor named %s found", param);

            return -1;

        }



        return fd;

    } else {

        return strtol(param, NULL, 0);

    }

}
