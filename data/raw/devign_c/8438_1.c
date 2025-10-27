static av_cold int init(AVFilterContext *ctx)

{

    SendCmdContext *sendcmd = ctx->priv;

    int ret, i, j;



    if (sendcmd->commands_filename && sendcmd->commands_str) {

        av_log(ctx, AV_LOG_ERROR,

               "Only one of the filename or commands options must be specified\n");

        return AVERROR(EINVAL);

    }



    if (sendcmd->commands_filename) {

        uint8_t *file_buf, *buf;

        size_t file_bufsize;

        ret = av_file_map(sendcmd->commands_filename,

                          &file_buf, &file_bufsize, 0, ctx);

        if (ret < 0)

            return ret;



        /* create a 0-terminated string based on the read file */

        buf = av_malloc(file_bufsize + 1);

        if (!buf) {

            av_file_unmap(file_buf, file_bufsize);

            return AVERROR(ENOMEM);

        }

        memcpy(buf, file_buf, file_bufsize);

        buf[file_bufsize] = 0;

        av_file_unmap(file_buf, file_bufsize);

        sendcmd->commands_str = buf;

    }



    if ((ret = parse_intervals(&sendcmd->intervals, &sendcmd->nb_intervals,

                               sendcmd->commands_str, ctx)) < 0)

        return ret;



    if (sendcmd->nb_intervals == 0) {

        av_log(ctx, AV_LOG_ERROR, "No commands\n");

        return AVERROR(EINVAL);

    }



    qsort(sendcmd->intervals, sendcmd->nb_intervals, sizeof(Interval), cmp_intervals);



    av_log(ctx, AV_LOG_DEBUG, "Parsed commands:\n");

    for (i = 0; i < sendcmd->nb_intervals; i++) {

        AVBPrint pbuf;

        Interval *interval = &sendcmd->intervals[i];

        av_log(ctx, AV_LOG_VERBOSE, "start_time:%f end_time:%f index:%d\n",

               (double)interval->start_ts/1000000, (double)interval->end_ts/1000000, interval->index);

        for (j = 0; j < interval->nb_commands; j++) {

            Command *cmd = &interval->commands[j];

            av_log(ctx, AV_LOG_VERBOSE,

                   "    [%s] target:%s command:%s arg:%s index:%d\n",

                   make_command_flags_str(&pbuf, cmd->flags), cmd->target, cmd->command, cmd->arg, cmd->index);

        }

    }



    return 0;

}
