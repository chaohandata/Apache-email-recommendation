import com.lucidworks.spark.rdd.SolrRDD

val sqlContext = spark.sqlContext
val options = Map(
  "collection" -> "lucidfind",
  "query"->"id:<69FABED6-6CD6-4202-BEB6-11DF4A5C355F@maprtech.com> OR id:<CAB+VZSDwZm=ui4hJwdJVUkpGdpcW+W3cpfbzDqxwt6dEEfze=A@mail.gmail.com> OR id:<3022044060950341124@unknownmsgid> OR id:<555BA422.6090708@apache.org> OR id:<CADT+BFH8FFbMgNmqws9w4xaFWE8PjXjm7DnABhmUre02VOpibg@mail.gmail.com> OR id:<CAJkA4MEq37yNTCmusU769sA9CiLv6xYj7BHDKOkd7-5DWDYoKw@mail.gmail.com> OR id:<CAKOFcwoFQh+6TzNWYc06ZM9VXFrUvyoXV+oCMAw0oWp_N-hpxw@mail.gmail.com> OR  id:<CADAN2-aFQ7Y2kOhTGZ746NtDwR07cMPiReEHMK3EvSBsxNJh0g@mail.gmail.com> OR id:<555EE763.20909@apache.org> OR id:<CA+dwJQBg4YHs45=X=zsG+aaFS0Xncz0KH+Bv=FoZ5bKRoR6saA@mail.gmail.com> OR id:<CADAN2-ZrGeLsK0wZfi-voUnj8XofKPnaLr3DC+pJ4iJ+OJDmNg@mail.gmail.com> OR id:<CAGXsTJd_7WQOgqu1m6zw-SVk8ewg2X-cvXpdEg6zLSR-r-TLVQ@mail.gmail.com> OR  id:<CADAN2-YZEKju5qkErF960uJJ9MsWCqNLJQ8noRtABbDMCgofsw@mail.gmail.com> OR id:<1433766158.2408.9.camel@goodgamestudios.com> OR id:<CA+dwJQBg4YHs45=X=zsG+aaFS0Xncz0KH+Bv=FoZ5bKRoR6saA@mail.gmail.com> OR id:<CAAowrT9vYvtXzrasLFPS14rvTZUTX88CXQeO6r0+H7VY2LrcWg@mail.gmail.com>",
  "zkhost" -> "localhost:9983/lwfusion/3.1.0/solr"
)
val rawDF = spark.read.format("solr").options(options).load