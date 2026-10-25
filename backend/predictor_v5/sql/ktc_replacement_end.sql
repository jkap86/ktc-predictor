SELECT h.format, h.ktc_id, k.sleeper_id, h.date, h.sf_value
  FROM ktc_value_history h
  JOIN ktc_values k USING (format, ktc_id)
 WHERE h.format = 'dynasty'
   AND k.sleeper_id IS NOT NULL
   AND h.sf_value > 0
   AND h.date BETWEEN '2026-12-28'::date AND '2027-01-10'::date
 ORDER BY h.format, h.ktc_id, h.date, k.sleeper_id, h.sf_value;
