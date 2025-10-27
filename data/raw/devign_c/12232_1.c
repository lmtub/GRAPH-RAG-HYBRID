static uint16List **host_memory_append_node(uint16List **node,

                                            unsigned long value)

{

     *node = g_malloc0(sizeof(**node));

     (*node)->value = value;

     return &(*node)->next;

}
