static void runstate_init(void)

{

    const RunStateTransition *p;



    memset(&runstate_valid_transitions, 0, sizeof(runstate_valid_transitions));



    for (p = &runstate_transitions_def[0]; p->from != RUN_STATE_MAX; p++) {

        runstate_valid_transitions[p->from][p->to] = true;

    }

}
