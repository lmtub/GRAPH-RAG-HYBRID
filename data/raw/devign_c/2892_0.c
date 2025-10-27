static void guest_file_init(void)

{

    QTAILQ_INIT(&guest_file_state.filehandles);

}
