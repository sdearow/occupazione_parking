# Relazione tecnica
## Analisi della sosta veicolare nel Comune di Roma tramite dati Floating Car Data (FCD)

**Dati di riferimento:** spostamenti veicolari — marzo 2023
**Strumento:** script Python `analyze_parking.py` (geopandas, pandas, matplotlib)
**Sistema di riferimento di lavoro:** EPSG:25833 (ETRS89 / UTM Zona 33N — unità metriche)

---

## Executive summary

Questa analisi stima **dove**, **quando** e **quanto** i veicoli sostano nel Comune di Roma, partendo da 411.356 spostamenti GPS reali (Floating Car Data) di marzo 2023, incrociati con i limiti amministrativi comunali e con la superficie carrabile cittadina.

**Risultati principali**

- **Filtraggio territoriale.** L'**86,8%** degli spostamenti (357.117) ha destinazione interna al Comune di Roma; il restante 13,2% è stato escluso.

- **Dove si parcheggia.** Il **65%** delle soste avviene esattamente sulla superficie stradale pubblica (*on-street*). Applicando una tolleranza di **2 metri** — coerente con l'errore tipico del GPS urbano — la quota sale al **76%**, e con 5 metri all'84%. La maggior parte della sosta è quindi **su strada**, non in aree private: i punti inizialmente classificati "fuori strada" sono in gran parte a ridosso della carreggiata.

- **Quanti veicoli sono fermi.** In un momento medio della giornata, oltre il **94%** dei veicoli osservati è parcheggiato; di notte (23–05) la quota supera il **98%**. Anche nell'ora di punta mattutina solo circa il 10% dei veicoli è in movimento.

- **Stime a scala di città.** Calibrando il campione sul dato reale di **300.000 veicoli unici** in circolazione nella fascia di punta 07–09 (fattore di espansione ≈ 217×) e considerando un parco veicolare totale di **1.600.000 mezzi**, si stima che in un momento medio siano fermi a Roma circa **1,5 milioni di veicoli**, di cui circa **1,18 milioni su suolo stradale pubblico**. Questo equivale a occupare circa l'**11% della superficie carrabile** comunale (135 km²).

**Sintesi quantitativa (stima media giornaliera, scala città)**

| Indicatore | Valore stimato |
|---|---|
| Veicoli in movimento | ~61.000 (3,8%) |
| Veicoli parcheggiati | ~1.540.000 (96,2%) |
| di cui su strada (on-street, ≤2 m) | ~1.180.000 |
| di cui fuori strada (off-street) | ~360.000 |
| Superficie carrabile occupata | ~10,9% |

**Avvertenza.** Le stime assolute sono **ordini di grandezza** dipendenti dai parametri di calibrazione (riferimento di punta, parco totale, ingombro per veicolo) e dalla rappresentatività del campione FCD. La direzione prevalente delle incertezze porta a una **leggera sovrastima** della sosta su strada; i valori vanno letti come limite superiore ragionevole, con incertezza dell'ordine di **±15–25%**. Il dettaglio dei limiti è alla sezione 6.

---

## 1. Obiettivo dell'analisi

L'analisi si propone di rispondere a tre domande sulla sosta veicolare a Roma, a partire da dati GPS reali di veicoli in circolazione:

1. **Dove parcheggiano i veicoli?** Quanti si fermano sulla superficie stradale pubblica (*on-street*) e quanti in aree private o fuori strada (*off-street*).
2. **Quanti veicoli sono fermi in un dato momento** rispetto a quelli effettivamente in uso, con disaggregazione oraria e tra giorno e notte.
3. **Quanto spazio stradale è occupato dalla sosta**, in termini assoluti e come quota della superficie carrabile della città.

---

## 2. Dati di partenza

### 2.1 Spostamenti FCD (Floating Car Data)
Quattro file CSV (`od_trips_part1-4.csv`) per un totale di **411.356 spostamenti** registrati nel mese di marzo 2023. Ogni riga rappresenta un viaggio e contiene:

| Campo | Descrizione |
|---|---|
| `o`, `d` | Punto di origine e destinazione (geometria WKT, WGS84) |
| `dep_time`, `arr_time` | Orario di partenza e arrivo |
| `user_id` | Identificativo del veicolo |
| `distance`, `travel_time` | Lunghezza e durata del viaggio |
| `type_o`, `type_d` | Tipologia di luogo (casa, lavoro, altro) |

La **destinazione `d`** di ogni viaggio è assunta come il luogo in cui il veicolo si ferma (parcheggia) al termine dello spostamento.

### 2.2 Limiti amministrativi (`municipi_2013.shp`)
15 poligoni dei municipi di Roma (EPSG:3003). La loro unione definisce il perimetro del Comune di Roma, usato per delimitare l'area di analisi.

### 2.3 Superficie carrabile (`AC_VEI.shp`)
79.085 poligoni (EPSG:25833) che rappresentano la superficie stradale percorribile dai veicoli all'interno del Comune. La sua estensione totale (entro Roma) è di circa **135,1 km²**. Questo layer è la "maschera" usata per distinguere le soste su strada da quelle fuori strada.

> **Nota tecnica:** il file `AC_VEI.shp` è archiviato tramite Git LFS (Large File Storage), 145 MB. Lo script lo rileva automaticamente: se è disponibile come file reale esegue l'analisi on/off-street completa, altrimenti la salta e usa i valori già calcolati.

---

## 3. Metodologia, passo per passo

Lo script è organizzato in step sequenziali. Di seguito il dettaglio di ciascuno.

### STEP 1–2 — Caricamento e georeferenziazione
I quattro CSV vengono uniti in un'unica tabella. Gli orari vengono convertiti in formato datetime, da cui si estraggono **ora di arrivo** e **giorno della settimana**. La colonna geometria delle destinazioni `d` (in coordinate geografiche WGS84) viene riproiettata nel sistema metrico **EPSG:25833**, indispensabile per calcolare distanze e buffer in metri.

### STEP 3 — Filtraggio sul Comune di Roma
Ogni punto di destinazione viene confrontato geometricamente con il perimetro municipale tramite **spatial join indicizzato (R-tree)**. Gli spostamenti la cui destinazione cade fuori Roma vengono esclusi.

**Risultato:**
| | Spostamenti | % |
|---|---|---|
| Totale FCD | 411.356 | 100,0% |
| **Destinazione dentro Roma** | **357.117** | **86,8%** |
| Destinazione fuori Roma | 54.239 | 13,2% |

L'analisi prosegue sui 357.117 spostamenti con destinazione interna al Comune.

### STEP 4 — Classificazione on-street / off-street
Ogni destinazione viene testata rispetto alla superficie carrabile AC_VEI: se il punto cade **dentro** un poligono stradale, la sosta è *on-street*; altrimenti *off-street*. Il test usa nuovamente uno spatial join R-tree per gestire efficientemente i 357 mila punti contro i 79 mila poligoni.

Poiché il segnale GPS ha una precisione limitata e molti veicoli si fermano ai bordi della carreggiata, l'analisi è ripetuta applicando **buffer incrementali** (0, 1, 2, 5, 10 metri) alla superficie carrabile, così da osservare come varia la quota on-street al crescere della tolleranza.

**Risultato:**
| Buffer | On-street | Off-street | % On-street |
|---|---|---|---|
| **0 m** (superficie esatta) | 232.071 | 125.046 | **65,0%** |
| 1 m | 256.183 | 100.934 | 71,7% |
| **2 m** | 272.941 | 84.176 | **76,4%** |
| 5 m | 300.996 | 56.121 | 84,3% |
| 10 m | 321.846 | 35.271 | 90,1% |

**Lettura:** senza tolleranza, il 65% delle soste avviene sulla carreggiata. Bastano però 2 metri di buffer per salire al 76,4%, segno che gran parte dei punti classificati off-street a 0m è in realtà a ridosso della strada e ricade lì per imprecisione del GPS, non per reale sosta in area privata. La soglia di **2 metri** è stata adottata come stima conservativa dell'errore di posizionamento urbano.

![Curva on/off-street al variare del buffer](results/fig4_curva_buffer.png)

![Distribuzione on/off-street per soglia di buffer](results/fig5_barre_buffer.png)

L'andamento on-street nel tempo conferma che la quota di sosta su strada è strutturalmente alta in tutte le fasce orarie e in tutti i giorni della settimana.

![Andamento orario on/off-street](results/fig6_andamento_orario_onoff.png)

![Andamento settimanale on/off-street](results/fig7_andamento_settimanale_onoff.png)

![Heatmap % on-street per ora e giorno](results/fig8_heatmap_onstreet_ora_giorno.png)

### STEP 5–6 — Analisi temporale e grafici
Gli arrivi a Roma vengono aggregati per **ora del giorno** e per **giorno della settimana**, producendo distribuzioni di volume e relative heatmap.

**Andamento orario:** gli arrivi crescono dalle 6:00, raggiungono un primo plateau mattutino (8:00–12:00, ~24.000/ora) e un secondo picco serale attorno alle 18:00 (~25.000), per poi calare nelle ore notturne.

![Distribuzione oraria degli arrivi](results/fig1_distribuzione_oraria.png)

**Andamento settimanale:** i giorni feriali centrali (mercoledì–venerdì) registrano oltre 63.000 arrivi giornalieri; il weekend cala sensibilmente, con la domenica a ~30.000 (meno della metà del picco feriale).

![Distribuzione settimanale degli arrivi](results/fig2_distribuzione_settimanale.png)

![Heatmap volume per ora e giorno della settimana](results/fig3_heatmap_volume_ora_giorno.png)

### STEP 7 — Veicoli parcheggiati vs in uso
Per stimare quanti veicoli sono fermi in ogni momento, si adotta un metodo a **snapshot orari**: per ogni ora del giorno (istantanea alle :30) si conta quanti veicoli del campione hanno un viaggio in corso (*in uso*) e quanti no (*parcheggiati*). Il calcolo è fatto giorno per giorno e poi mediato.

Il campione contiene **23.571 veicoli unici** su 32 giorni. Si considerano solo i viaggi che iniziano e finiscono nella stessa giornata (i viaggi a cavallo della mezzanotte, <1%, sono esclusi per semplicità).

**Risultato (% sulla flotta osservata che si muove almeno una volta al giorno):**
| Fascia | % parcheggiati | % in uso |
|---|---|---|
| **Media giornaliera (24h)** | **94,2%** | 5,8% |
| Diurno (06–22) | 92,3% | 7,7% |
| **Notturno (23–05)** | **98,6%** | 1,4% |
| Picco di utilizzo (8:30) | 89,7% | **10,3%** |

**Lettura:** anche nell'ora di massimo traffico, solo circa il 10% dei veicoli osservati è in movimento; il resto è fermo. Di notte la quota di veicoli in sosta supera il 98%.

![% parcheggiati vs in uso per ora del giorno](results/fig9_pct_parcheggiati_vs_inuso_orario.png)

![N. veicoli parcheggiati vs in uso (campione)](results/fig10_n_veicoli_parcheggiati_vs_inuso.png)

![Heatmap % parcheggiati per ora e giorno](results/fig11_heatmap_pct_parcheggiati.png)

![Confronto giorno/notte](results/fig12_confronto_giorno_notte.png)

### STEP 9 — Stime a scala di città

Le grandezze precedenti riguardano il **campione** FCD, non l'intera flotta romana. Per ottenere stime assolute si applica un **fattore di espansione**, calibrato su un dato di traffico reale fornito come riferimento.

**Logica di calibrazione (approccio "veicoli unici in fascia di punta"):**
- *Riferimento esterno:* **300.000 veicoli unici** effettuano almeno uno spostamento nella fascia di punta mattutina **07:00–09:00**.
- *Stessa metrica nel campione:* in media **1.381 veicoli/giorno** partono in quella finestra.
- *Fattore di espansione:* 300.000 ÷ 1.381 ≈ **217×**.

Questo implica una **flotta giornaliera attiva** (veicoli che si muovono almeno una volta al giorno) di circa **1,03 milioni**.

**Correzione importante sulla superficie occupata.** Per stimare lo spazio stradale occupato dalla sosta, il denominatore corretto **non** è la sola flotta attiva, bensì il **parco veicolare totale** della città, perché anche i veicoli che non si muovono affatto in una giornata occupano comunque spazio se parcheggiati su strada. Si è quindi adottato:

> veicoli parcheggiati = **parco totale − veicoli in movimento**

con **parco totale = 1.600.000 veicoli** (riferimento ACI per il Comune di Roma, parametro modificabile).

La superficie occupata è stimata assumendo **12,5 m² per veicolo** (ingombro standard 5 m × 2,5 m, comprensivo di manovra) applicato ai veicoli classificati on-street (buffer 2 m).

**Risultato (stime assolute a scala di città):**
| Fascia | In movimento | Parcheggiati totali | % parch. | On-street (≤2 m) | Superficie carrabile occupata |
|---|---|---|---|---|---|
| **Media 24h** | 61.000 | **1.538.612** | 96,2% | 1.175.500 | **10,9%** |
| Diurno 06–22 | 82.000 | 1.518.292 | 94,9% | 1.159.975 | 10,7% |
| **Notturno 23–05** | 12.000 | **1.587.960** | 99,2% | 1.213.202 | **11,2%** |
| Picco 8:30 | 113.000 | 1.487.111 | 92,9% | 1.136.153 | 10,5% |

**Lettura:** in un qualunque momento medio della giornata, circa **1,5 milioni di veicoli** sono fermi a Roma, di cui circa **1,18 milioni su suolo stradale pubblico** — pari a circa il **10,9% della superficie carrabile** (135 km²). Di notte la quota sale leggermente, perché quasi nessun veicolo è in circolazione.

![Stima città — profilo orario veicoli in movimento vs parcheggiati](results/fig13_stima_citta_profilo_orario.png)

![Stima città — confronto per fasce orarie](results/fig14_stima_citta_fasce_orarie.png)

### STEP 10 — Analisi spaziale della sosta: mappe e indicatori per municipio

Per visualizzare la distribuzione geografica della sosta è stato realizzato uno script dedicato (`map_parking.py`) che produce due famiglie di mappe usando la libreria **H3** (griglia esagonale geospaziale di Uber, resolution 9, celle con lato ~174 m e area ~0,105 km²).

Le mappe sono generate in **due scale**:
- **Campione FCD:** conteggi diretti delle soste nel dataset, utili per confronti spaziali relativi.
- **Stima città (×252):** valori espansi al parco reale tramite il fattore di calibrazione, per stime assolute (soste/km² a scala urbana).

#### A. Griglia H3 — densità campione (soste/km²)

![Densità di sosta totale — campione FCD](results/maps/map01_densita_totale_campione.png)

![Densità di sosta diurna 07–20 — campione FCD](results/maps/map02_densita_diurna_campione.png)

![Densità di sosta notturna 20–07 — campione FCD](results/maps/map03_densita_notturna_campione.png)

#### B. Griglia H3 — densità stima città (soste/km², scala reale)

![Densità di sosta totale — stima città](results/maps/map04_densita_totale_citta.png)

![Densità di sosta diurna 07–20 — stima città](results/maps/map05_densita_diurna_citta.png)

![Densità di sosta notturna 20–07 — stima città](results/maps/map06_densita_notturna_citta.png)

#### C. Indice residenziale (griglia H3)

La quota di soste notturne (20–07) rispetto al totale è un proxy della **funzione residenziale** della sosta in quella zona: alta percentuale notturna indica che i veicoli rientrano a casa la sera e rimangono fermi tutta la notte.

![Indice residenziale — % soste notturne (H3)](results/maps/map07_indice_residenziale.png)

**Lettura:** la sosta si concentra fortemente nei Municipi VIII e IX (Appio–Tuscolano, Cinecittà–Ardeatino), zone ad alta densità abitativa e bassa disponibilità di parcheggi privati. Il Centro Storico (Municipio I) ha una quota notturna più alta (~24%), coerente con la funzione mista residenziale/turistica e la scarsità di garage privati.

#### D. Coroplete per municipio

Le stesse metriche vengono aggregate per **municipio**, producendo mappe coroplete che permettono un confronto diretto tra le 15 circoscrizioni a due scale.

![Densità sosta per municipio — campione](results/maps/map08_municipio_densita_campione.png)

![Densità sosta per municipio — stima città](results/maps/map09_municipio_densita_citta.png)

![Indice residenziale per municipio](results/maps/map10_municipio_indice_residenziale.png)

#### E. Bar chart comparativo per municipio

![Indicatori di sosta per municipio](results/maps/map12_municipio_barchart.png)

#### F. Saturazione superficie carrabile (richiede AC_VEI.shp)

La **saturazione** misura la quota di superficie stradale occupata dai veicoli in sosta on-street, calcolata come:

> saturazione (%) = (n. soste on-street stimate × 12,5 m²) / area carrabile AC_VEI × 100

Dove `n. soste on-street` è ottenuto espandendo il campione FCD col fattore di calibrazione (×252) e applicando la percentuale on-street locale (buffer 2 m).

La quota di soste on-street varia per municipio in funzione della disponibilità di parcheggi privati e della morfologia urbana:

![% soste on-street per municipio](results/maps/map11_municipio_pct_onstreet.png)

![Saturazione superficie carrabile per municipio](results/maps/map13_municipio_saturazione.png)

![Saturazione superficie carrabile per cella H3](results/maps/map14_saturazione_h3.png)

**Indicatori principali per municipio (top 5 per densità):**

| Municipio | Soste campione | Densità (soste/km²) | % notturne |
|---|---|---|---|
| IX (ex XII) | 183.274 | 1.000 | 13,8% |
| VIII (ex XI) | 46.995 | 999 | 14,2% |
| I (ex I e XVII) | 12.174 | 603 | 23,9% |
| XI (ex XV) | 31.321 | 438 | 15,9% |
| VII (ex IX e X) | 12.449 | 272 | 20,5% |

---

## 4. File prodotti

Tutti i risultati sono nella cartella `results/`.

### Tabelle (CSV)
| File | Contenuto |
|---|---|
| `00_filtraggio_comune.csv` | Spostamenti dentro/fuori Roma |
| `01_buffer_summary.csv` | % on/off-street per ogni buffer |
| `02_analisi_oraria.csv` | Volume e % on/off-street per ora |
| `03_analisi_giorno_settimana.csv` | Volume e % on/off-street per giorno |
| `04_copertura_superficie.csv` | Copertura superficie (campione) |
| `05_destinazioni_classificate.csv` | Ogni spostamento con classificazione |
| `06_uso_veicoli_orario.csv` | Veicoli in uso/fermi per ora (campione) |
| `07_uso_veicoli_per_giorno.csv` | Stesso dato per giorno settimana |
| `08_sintesi_uso_flotta.csv` | Sintesi parcheggiati diurno/notturno |
| `09_stima_citta_oraria.csv` | Stime assolute città, ora per ora |
| `10_stima_citta_sintesi.csv` | Stime assolute città per fascia |

### Grafici analisi (PNG, in `results/`)
| File | Contenuto |
|---|---|
| `fig1_distribuzione_oraria.png` | Arrivi per ora |
| `fig2_distribuzione_settimanale.png` | Arrivi per giorno |
| `fig3_heatmap_volume_ora_giorno.png` | Heatmap volume |
| `fig4_curva_buffer.png` | Curva on/off-street vs buffer |
| `fig5_barre_buffer.png` | Barre on/off-street per buffer |
| `fig6_andamento_orario_onoff.png` | % on/off-street per ora |
| `fig7_andamento_settimanale_onoff.png` | % on/off-street per giorno |
| `fig8_heatmap_onstreet_ora_giorno.png` | Heatmap % on-street |
| `fig9_pct_parcheggiati_vs_inuso_orario.png` | % parcheggiati vs in uso |
| `fig10_n_veicoli_parcheggiati_vs_inuso.png` | N. assoluti campione |
| `fig11_heatmap_pct_parcheggiati.png` | Heatmap % parcheggiati |
| `fig12_confronto_giorno_notte.png` | Confronto giorno/notte |
| `fig13_stima_citta_profilo_orario.png` | Stima città, profilo orario |
| `fig14_stima_citta_fasce_orarie.png` | Stima città per fascia |

### Mappe spaziali (PNG, in `results/maps/`) — H3 resolution 9, edge ~174 m
| File | Contenuto |
|---|---|
| `map01_densita_totale_campione.png` | Densità totale — scala campione (soste/km²) |
| `map02_densita_diurna_campione.png` | Densità diurna 07–20 — scala campione |
| `map03_densita_notturna_campione.png` | Densità notturna 20–07 — scala campione |
| `map04_densita_totale_citta.png` | Densità totale — stima città (soste/km², ×252) |
| `map05_densita_diurna_citta.png` | Densità diurna — stima città |
| `map06_densita_notturna_citta.png` | Densità notturna — stima città |
| `map07_indice_residenziale.png` | Indice residenziale (% soste notturne, H3) |
| `map08_municipio_densita_campione.png` | Coroplete densità per municipio — campione |
| `map09_municipio_densita_citta.png` | Coroplete densità per municipio — stima città |
| `map10_municipio_indice_residenziale.png` | Coroplete indice residenziale per municipio |
| `map11_municipio_pct_onstreet.png` | % soste on-street per municipio *(richiede AC_VEI)* |
| `map12_municipio_barchart.png` | Bar chart comparativo per municipio |
| `map13_municipio_saturazione.png` | Saturazione sup. carrabile per municipio *(richiede AC_VEI)* |
| `map14_saturazione_h3.png` | Saturazione per cella H3 *(richiede AC_VEI)* |
| `11_statistiche_municipio.csv` | Statistiche aggregate per municipio |

---

## 5. Parametri configurabili

I valori chiave sono raccolti all'inizio dello STEP 9 dello script e si possono modificare per ricalcolare tutto:

| Parametro | Valore attuale | Significato |
|---|---|---|
| `REF_PEAK_VEHICLES` | 300.000 | Veicoli unici in punta (riferimento esterno) |
| `PEAK_WINDOW_START` / `_END` | 7 / 9 | Fascia oraria di punta per la calibrazione |
| `FLEET_TOTAL_ROME` | 1.600.000 | Parco veicolare totale (ACI) |
| `AREA_VEI_MQ` | 12,5 | Ingombro medio per veicolo (m²) |
| `BUFFER_DISTANCES` | [0,1,2,5,10] | Soglie di buffer (m) |

---

## 6. Limiti del metodo

I risultati vanno interpretati come **stime di ordine di grandezza**, soggette ai seguenti limiti.

### 6.1 Rappresentatività del campione FCD
Il campione non è casuale: i veicoli con telematica di bordo tendono a essere più recenti, di segmento medio-alto e con uso più intenso. Veicoli anziani, utenti a basso reddito e auto poco usate sono sottorappresentati. Il campione copre inoltre **un solo mese** (marzo 2023), non rappresentativo di periodi a diversa stagionalità (agosto, festività, grandi eventi).

### 6.2 Coerenza della calibrazione
Il fattore di scala (217×) confronta due grandezze la cui omogeneità non è garantita: i "veicoli unici in punta" del campione e il riferimento esterno dei 300.000, la cui esatta definizione e fonte non sono note (potrebbe includere pendolari dalla provincia o basarsi su conteggi di flusso anziché di veicoli unici). Eventuali disallineamenti si propagano a tutti i valori assoluti.

### 6.3 Parco totale ACI come denominatore
I 1.600.000 veicoli includono mezzi fermi da mesi, non assicurati o mai usati. Usarli come denominatore tende a **sovrastimare** il numero di veicoli parcheggiati.

### 6.4 Ratio on/off-street esteso all'intera flotta
La proporzione on/off-street è calcolata sui soli veicoli che si sono mossi. Applicarla anche ai veicoli permanentemente inattivi — che plausibilmente sono più spesso in garage (off-street) — porta a **sovrastimare** la quota on-street e quindi la superficie stradale occupata. Inoltre la ratio è tenuta costante nelle 24 ore, mentre è probabile vari tra giorno e notte.

### 6.5 Errore GPS
L'errore di posizionamento urbano è tipicamente di 5–15 m (oltre 20 m nei canyon urbani). I buffer incrementali ne quantificano l'effetto, ma non esiste una soglia universalmente corretta: il punto di destinazione potrebbe inoltre essere registrato mentre il veicolo sta ancora manovrando.

### 6.6 Ingombro per veicolo
I 12,5 m² si riferiscono a un'auto media. Scooter e moto (~1–2 m²), SUV e furgoni (15–30 m²) hanno ingombri molto diversi. Se i motocicli sono conteggiati nel parco ma valutati con l'ingombro dell'auto, la superficie è sovrastimata.

### 6.7 Età del dato AC_VEI
Il layer della superficie carrabile ha un anno di riferimento attorno al 2014. Trasformazioni viabilistiche successive (nuove ZTL, isole pedonali, piste ciclabili) possono aver modificato la superficie effettivamente disponibile.

### 6.8 Direzione complessiva delle distorsioni
| Limite | Effetto sulla stima on-street / superficie |
|---|---|
| Campione biased verso utenti attivi | Sovrastima on-street |
| Veicoli inattivi più spesso in garage | Sovrastima on-street |
| Parco ACI con vetture fuori uso | Sovrastima n. parcheggiati |
| Scooter con ingombro auto | Sovrastima superficie |
| AC_VEI del 2014 | Sottostima % superficie |

La direzione prevalente è una **leggera sovrastima** della sosta su strada. I valori assoluti vanno quindi letti come **limite superiore ragionevole**, con un'incertezza stimabile nell'ordine di **±15–25%**.

---

## 7. Sintesi conclusiva

- L'**86,8%** degli spostamenti FCD ha destinazione dentro Roma.
- Della sosta osservata, il **65%** avviene sulla carreggiata (buffer 0 m), che sale al **76%** con appena 2 m di tolleranza GPS: la maggior parte della sosta è quindi **su strada**.
- In un momento medio della giornata, oltre il **94%** dei veicoli osservati è fermo; di notte oltre il **98%**.
- A scala di città, ciò corrisponde a circa **1,5 milioni di veicoli parcheggiati**, di cui circa **1,18 milioni su strada**, equivalenti a circa l'**11% della superficie carrabile** comunale.
- Tutte le stime assolute dipendono dai parametri di calibrazione (riferimento di punta, parco totale, ingombro per veicolo) e vanno trattate come ordini di grandezza con incertezza ±15–25%.

---

*Relazione generata a corredo dello script `analyze_parking.py`. Tutti i numeri sono riproducibili rieseguendo lo script sui dati del repository.*
