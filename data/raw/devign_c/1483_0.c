static int is_async_return(const QObject *data)

{

    if (data && qobject_type(data) == QTYPE_QDICT) {

        return qdict_haskey(qobject_to_qdict(data), "__mon_async");

    }



    return 0;

}
