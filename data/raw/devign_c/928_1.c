void pcmcia_info(Monitor *mon, const QDict *qdict)

{

    struct pcmcia_socket_entry_s *iter;



    if (!pcmcia_sockets)

        monitor_printf(mon, "No PCMCIA sockets\n");



    for (iter = pcmcia_sockets; iter; iter = iter->next)

        monitor_printf(mon, "%s: %s\n", iter->socket->slot_string,

                       iter->socket->attached ? iter->socket->card_string :

                       "Empty");

}
