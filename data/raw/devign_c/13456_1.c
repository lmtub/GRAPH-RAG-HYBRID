void sh_intc_register_sources(struct intc_desc *desc,

			      struct intc_vect *vectors,

			      int nr_vectors,

			      struct intc_group *groups,

			      int nr_groups)

{

    unsigned int i, k;

    struct intc_source *s;



    for (i = 0; i < nr_vectors; i++) {

	struct intc_vect *vect = vectors + i;



	sh_intc_register_source(desc, vect->enum_id, groups, nr_groups);

	s = sh_intc_source(desc, vect->enum_id);

	if (s)

	    s->vect = vect->vect;



#ifdef DEBUG_INTC_SOURCES

	printf("sh_intc: registered source %d -> 0x%04x (%d/%d)\n",

	       vect->enum_id, s->vect, s->enable_count, s->enable_max);

#endif

    }



    if (groups) {

        for (i = 0; i < nr_groups; i++) {

	    struct intc_group *gr = groups + i;



	    s = sh_intc_source(desc, gr->enum_id);

	    s->next_enum_id = gr->enum_ids[0];



	    for (k = 1; k < ARRAY_SIZE(gr->enum_ids); k++) {

                if (!gr->enum_ids[k])

                    continue;



		s = sh_intc_source(desc, gr->enum_ids[k - 1]);

		s->next_enum_id = gr->enum_ids[k];

	    }



#ifdef DEBUG_INTC_SOURCES

	    printf("sh_intc: registered group %d (%d/%d)\n",

		   gr->enum_id, s->enable_count, s->enable_max);

#endif

	}

    }

}
