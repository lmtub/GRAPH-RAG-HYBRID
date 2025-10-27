void gdb_exit(CPUState *env, int code)

{

  GDBState *s;

  char buf[4];



  s = &gdbserver_state;

  if (gdbserver_fd < 0 || s->fd < 0)

    return;



  snprintf(buf, sizeof(buf), "W%02x", code);

  put_packet(s, buf);

}
