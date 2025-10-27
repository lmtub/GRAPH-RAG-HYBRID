int gdbserver_start(const char *port)

{

    GDBState *s;

    char gdbstub_port_name[128];

    int port_num;

    char *p;

    CharDriverState *chr;



    if (!port || !*port)

      return -1;



    port_num = strtol(port, &p, 10);

    if (*p == 0) {

        /* A numeric value is interpreted as a port number.  */

        snprintf(gdbstub_port_name, sizeof(gdbstub_port_name),

                 "tcp::%d,nowait,nodelay,server", port_num);

        port = gdbstub_port_name;

    }



    chr = qemu_chr_open("gdb", port);

    if (!chr)

        return -1;



    s = qemu_mallocz(sizeof(GDBState));

    if (!s) {

        return -1;

    }

    s->env = first_cpu; /* XXX: allow to change CPU */

    s->chr = chr;

    qemu_chr_add_handlers(chr, gdb_chr_can_receive, gdb_chr_receive,

                          gdb_chr_event, s);

    qemu_add_vm_stop_handler(gdb_vm_stopped, s);

    return 0;

}
