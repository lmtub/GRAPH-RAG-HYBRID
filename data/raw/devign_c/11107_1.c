MigrationCapabilityStatusList *qmp_query_migrate_capabilities(Error **errp)

{

    MigrationCapabilityStatusList *head = NULL;

    MigrationCapabilityStatusList *caps;

    MigrationState *s = migrate_get_current();

    int i;




    for (i = 0; i < MIGRATION_CAPABILITY_MAX; i++) {

        if (head == NULL) {

            head = g_malloc0(sizeof(*caps));

            caps = head;

        } else {

            caps->next = g_malloc0(sizeof(*caps));

            caps = caps->next;

        }

        caps->value =

            g_malloc(sizeof(*caps->value));

        caps->value->capability = i;

        caps->value->state = s->enabled_capabilities[i];

    }



    return head;

}