int main(int argc, char **argv)

{

    int ret = EXIT_SUCCESS;

    GAState *s = g_new0(GAState, 1);

    GAConfig *config = g_new0(GAConfig, 1);



    config->log_level = G_LOG_LEVEL_ERROR | G_LOG_LEVEL_CRITICAL;



    module_call_init(MODULE_INIT_QAPI);



    init_dfl_pathnames();

    config_load(config);

    config_parse(config, argc, argv);



    if (config->pid_filepath == NULL) {

        config->pid_filepath = g_strdup(dfl_pathnames.pidfile);




    if (config->state_dir == NULL) {

        config->state_dir = g_strdup(dfl_pathnames.state_dir);




    if (config->method == NULL) {

        config->method = g_strdup("virtio-serial");




    if (config->channel_path == NULL) {

        if (strcmp(config->method, "virtio-serial") == 0) {

            /* try the default path for the virtio-serial port */

            config->channel_path = g_strdup(QGA_VIRTIO_PATH_DEFAULT);

        } else if (strcmp(config->method, "isa-serial") == 0) {

            /* try the default path for the serial port - COM1 */

            config->channel_path = g_strdup(QGA_SERIAL_PATH_DEFAULT);

        } else {

            g_critical("must specify a path for this channel");

            ret = EXIT_FAILURE;

            goto end;





    s->log_level = config->log_level;

    s->log_file = stderr;

#ifdef CONFIG_FSFREEZE

    s->fsfreeze_hook = config->fsfreeze_hook;

#endif

    s->pstate_filepath = g_strdup_printf("%s/qga.state", config->state_dir);

    s->state_filepath_isfrozen = g_strdup_printf("%s/qga.state.isfrozen",

                                                 config->state_dir);

    s->frozen = check_is_frozen(s);



    if (config->dumpconf) {

        config_dump(config);

        goto end;




    ret = run_agent(s, config);



end:

    if (s->command_state) {

        ga_command_state_cleanup_all(s->command_state);

        ga_command_state_free(s->command_state);

        json_message_parser_destroy(&s->parser);


    if (s->channel) {

        ga_channel_free(s->channel);


    g_free(s->pstate_filepath);

    g_free(s->state_filepath_isfrozen);



    if (config->daemonize) {

        unlink(config->pid_filepath);




    config_free(config);







    return ret;
