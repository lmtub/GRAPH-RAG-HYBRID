static MegasasCmd *megasas_next_frame(MegasasState *s,

    hwaddr frame)

{

    MegasasCmd *cmd = NULL;

    int num = 0, index;



    cmd = megasas_lookup_frame(s, frame);

    if (cmd) {

        trace_megasas_qf_found(cmd->index, cmd->pa);

        return cmd;

    }

    index = s->reply_queue_head;

    num = 0;

    while (num < s->fw_cmds) {

        if (!s->frames[index].pa) {

            cmd = &s->frames[index];

            break;

        }

        index = megasas_next_index(s, index, s->fw_cmds);

        num++;

    }

    if (!cmd) {

        trace_megasas_qf_failed(frame);

    }

    trace_megasas_qf_new(index, cmd);

    return cmd;

}
