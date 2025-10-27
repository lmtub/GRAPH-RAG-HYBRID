static inline int media_present(IDEState *s)

{

    return (s->nb_sectors > 0);

}
