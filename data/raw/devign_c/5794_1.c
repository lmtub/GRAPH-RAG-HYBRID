static void usage(const char *cmd)

{

    printf(

"Usage: %s [-m <method> -p <path>] [<options>]\n"

"QEMU Guest Agent %s\n"

"\n"

"  -m, --method      transport method: one of unix-listen, virtio-serial, or\n"

"                    isa-serial (virtio-serial is the default)\n"

"  -p, --path        device/socket path (the default for virtio-serial is:\n"

"                    %s)\n"

"  -l, --logfile     set logfile path, logs to stderr by default\n"

"  -f, --pidfile     specify pidfile (default is %s)\n"

"  -v, --verbose     log extra debugging information\n"

"  -V, --version     print version information and exit\n"

"  -d, --daemonize   become a daemon\n"

#ifdef _WIN32

"  -s, --service     service commands: install, uninstall\n"

#endif

"  -b, --blacklist   comma-separated list of RPCs to disable (no spaces, \"?\"\n"

"                    to list available RPCs)\n"

"  -h, --help        display this help and exit\n"

"\n"

"Report bugs to <mdroth@linux.vnet.ibm.com>\n"

    , cmd, QGA_VERSION, QGA_VIRTIO_PATH_DEFAULT, QGA_PIDFILE_DEFAULT);

}
