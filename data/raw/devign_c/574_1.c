CommandInfoList *qmp_query_commands(Error **errp)

{

    CommandInfoList *list = NULL;



    qmp_for_each_command(&qmp_commands, query_commands_cb, &list);



    return list;

}
