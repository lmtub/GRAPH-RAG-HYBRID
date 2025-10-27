static QDictEntry *qdict_find(const QDict *qdict,

                              const char *key, unsigned int hash)

{

    QDictEntry *entry;



    LIST_FOREACH(entry, &qdict->table[hash], next)

        if (!strcmp(entry->key, key))

            return entry;



    return NULL;

}
