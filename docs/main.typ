#import "@preview/arkheion:0.1.2": arkheion, arkheion-appendices

#show: arkheion.with(
  title: "Facial Emotion Recognition: uno studio comparativo tra una Custom CNN e una fine-tuned Xception",
  authors: (
    (name: "Daniele Lurani", email: "60/73/65357", affiliation: "Università degli Studi di Cagliari"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: "Il riconoscimento automatico delle espressioni facciali (FER) raggiunge accuracy
elevate quando training e valutazione provengono dallo stesso dataset, mentre la
robustezza al cambio di dominio resta meno documentata. Questo lavoro confronta due approcci sulle 7 classi emotive standard, una rete convoluzionale residuale addestrata da zero e una Xception pre-addestrata su ImageNet adattata tramite fine-tuning. Per Xception sono state provate tre strategie, che si differenziano per la presenza di una fase di warm-up e per quanta parte della rete viene riaddestrata, il 24,6% dei parametri oppure il 99,5%. Tutti i modelli sono addestrati solo su FER-2013 e valutati sia sul relativo test set sia, senza alcun riadattamento, su FANE. La selezione avviene su uno split di validation ricavato dal training set, così che entrambi i test set restino inutilizzati fino alla valutazione finale. Poiché nei due dataset le classi sono rappresentate in proporzioni molto diverse, i risultati sono riportati in accuracy, macro-F1 e weighted F1. Il fine-tuning completo è il modello migliore su entrambi i dataset e supera la rete addestrata da zero di 4,74 punti percentuali di accuracy su FER-2013 (IC 95% ±1,57) e di 3,00 punti su FANE (±1,11). Il fine-tuning parziale resta invece sotto la rete custom, quindi il vantaggio dei pesi pre-addestrati emerge solo quando gran parte della rete è libera di adattarsi. La fase di warm-up non produce differenze apprezzabili. Il divario di generalizzazione però non cambia. La differenza fra i due gap di accuracy è di 1,74 punti (±1,93), statisticamente indistinguibile da zero, e la macro-F1 cala del 43,5% per la rete custom contro il 45,4% di Xception. Il pre-addestramento su ImageNet migliora quindi le prestazioni assolute ma non la robustezza al cambio di dominio, che dipende dalle differenze fra i due dataset più che dall'architettura.
",
  date: "4 Settembre, 2026",
)
#set cite(style: "chicago-author-date")
#show link: underline

= Introduzione

== Contesto e motivazione

Il riconoscimento automatico delle espressioni facciali, indicato con la sigla
FER (Facial Emotion Recognition), consiste nell'assegnare a un'immagine di un
volto l'emozione che quel volto sta esprimendo. Dal punto di vista informatico è
un problema di classificazione supervisionata di immagini. L'insieme di
categorie più diffuso è quello delle emozioni di base, cioè rabbia, disgusto,
paura, felicità, tristezza e sorpresa, a cui si aggiunge la classe neutra per i
volti che non esprimono un'emozione marcata. Sono le sette classi adottate dai
due dataset impiegati in questo lavoro.
L'interesse per il FER nasce dal fatto che il volto è uno dei canali principali
con cui le persone comunicano il proprio stato emotivo. Un sistema capace di
interpretarlo trova impiego nell'interazione uomo-macchina, per costruire
interfacce che si adattano a come l'utente sta reagendo, e nell'analisi
comportamentale, per valutare la reazione di un pubblico o l'attenzione di chi
guida. Un terzo ambito è quello assistenziale, con strumenti pensati per chi ha
difficoltà a interpretare le espressioni altrui.

Per lungo tempo il problema è stato affrontato estraendo dalle immagini
caratteristiche progettate a mano, come i descrittori di texture o la posizione
di alcuni punti di riferimento del volto, per poi passarle a un classificatore.
Le reti neurali convoluzionali hanno cambiato questo modo di procedere, perché
imparano direttamente dai dati quali caratteristiche siano utili senza che
debbano essere definite in anticipo. È su questa famiglia di modelli che si
concentra il lavoro.

== Il problema della generalizzazione

La prassi più comune per valutare un modello di FER consiste nel dividere un
dataset in una parte per il training e una per il test, addestrare sulla prima e
riportare l'accuracy ottenuta sulla seconda. Il test set non viene mai visto
durante il training, quindi la misura sembra una stima onesta della bontà del
modello. Il punto debole è che le due parti provengono dalla stessa raccolta di
immagini.

Le immagini di un dataset condividono quasi sempre caratteristiche che dipendono
da come sono state raccolte, per esempio la risoluzione, il modo in cui il volto
è inquadrato, l'illuminazione, il tipo di persone fotografate e i criteri con cui
sono state assegnate le etichette. Un modello può imparare queste regolarità e
sfruttarle per rispondere correttamente sul test set, anche quando non hanno
nulla a che vedere con l'espressione in sé. L'accuracy descrive allora quanto il
modello funzioni su quel particolare dataset, non quanto sappia riconoscere le
espressioni in generale.

La differenza emerge quando il modello viene applicato a immagini di un'altra
fonte. La situazione in cui i dati su cui il modello lavora seguono una
distribuzione diversa da quella dei dati di training viene chiamata domain shift
e comporta di norma un calo delle prestazioni. È esattamente la condizione di un
sistema di FER usato in un'applicazione reale.
Per quantificare quel calo serve una valutazione cross-dataset, cioè addestrare
su una raccolta e valutare su una seconda raccolta indipendente, senza alcun
riadattamento. È l'impostazione adottata in questo lavoro.

== Obiettivi del lavoro

Il lavoro risponde a quattro domande, che ne definiscono anche la struttura.

La prima riguarda il confronto fra i due approcci più diffusi per costruire un
classificatore di espressioni. Una rete convoluzionale progettata e addestrata da
zero riesce a competere con un modello di grandi dimensioni pre-addestrato su un
altro compito e poi adattato tramite transfer learning? È una domanda pratica,
perché il transfer learning viene spesso adottato come scelta predefinita, nel
presupposto che partire da pesi già addestrati sia comunque vantaggioso.

La seconda domanda nasce dal fatto che il transfer learning non consiste in una sola tecnica. Si può riaddestrare solo il classificatore finale, oppure gli ultimi
blocchi convolutivi, oppure la rete intera, e la scelta cambia quanti parametri
sono liberi di adattarsi al nuovo compito. Quanto conviene riaddestrare? La
risposta si rivela decisiva, al punto che la risposta alla prima domanda cambia a
seconda di come si imposta il fine-tuning.

La terza domanda riguarda l'entità del calo introdotto dal domain shift. Quanto
perde un modello addestrato su FER-2013 quando viene valutato su un dataset
raccolto in modo indipendente? Serve a dare un ordine di grandezza concreto al
problema descritto sopra.

La quarta unisce le altre. Il pre-addestramento su ImageNet, che espone la rete a
un numero enorme di immagini molto varie, la rende più robusta al cambio di
dominio e riduce quindi quel calo? È l'ipotesi intuitiva, ma i risultati mostrano
che non trova conferma.

A queste si affiancano due verifiche minori, l'effetto di una fase iniziale di
warm-up nel fine-tuning e quello del batch size, discusse insieme ai risultati.

= Stato dell'arte

FER-2013 è uno dei dataset più usati per il riconoscimento delle espressioni
facciali, e questo rende disponibili alcuni valori di riferimento con cui
confrontare i risultati ottenuti.

Il primo è l'accuracy raggiunta da annotatori umani sullo stesso insieme di
immagini, stimata intorno al 65%. Il secondo è il risultato dei modelli migliori
proposti in letteratura, che si collocano poco sopra il 70%.

Per capire quanto siano alti serve sapere da dove si parte. Con sette classi una
scelta casuale uniforme darebbe circa il 14%, ma il dataset è sbilanciato e un
classificatore che rispondesse sempre con la classe più frequente, happy,
otterrebbe già il 24,7% sul test set. È questo il pavimento realistico.
L'intervallo utile va quindi da circa il 25% a poco più del 70%, ed è più stretto
di quanto suggerisca il numero delle classi.

Il motivo di un tetto così basso va cercato nella qualità dei dati. Le immagini
sono state raccolte in modo automatico e non sono state controllate una per una,
quindi molte espressioni risultano ambigue anche per una persona e in alcuni casi
il volto non è nemmeno presente. Il limite non sta tanto nei modelli quanto nel
dataset.

Va infine precisato che entrambi i riferimenti valgono per una valutazione
condotta sullo stesso dataset usato per il training. Per la parte cross-dataset
di questo lavoro non esistono riferimenti altrettanto consolidati, e il divario
misurato fra FER-2013 e FANE va quindi letto in termini assoluti, come misura del
calo, senza pretendere di confrontarlo con valori di letteratura.

= Dati

== I due dataset

L'addestramento è stato svolto su FER-2013, che mette a disposizione 28.709 immagini per l'addestramento e 7.178 per il test, già suddivise dagli autori. Le immagini sono in scala di grigi e hanno una risoluzione di 48 per 48 pixel, con il volto che occupa quasi interamente il riquadro. Le sette classi sono rabbia, disgusto, paura, felicità, neutro, tristezza e sorpresa.

La valutazione fuori dominio è stata condotta su FANE, che copre le stesse sette categorie ma è molto diverso nella forma. Le immagini sono a colori, hanno risoluzione maggiore e non seguono un ritaglio uniforme. Il dataset contiene in realtà nove cartelle, perché include anche le classi confused e shy, che sono state escluse dall'analisi. La ragione è semplice, dato che un modello addestrato su FER-2013 conosce soltanto sette categorie e non ha alcun modo di prevedere le altre due. Tenerle avrebbe prodotto errori sistematici che non dipendono dal cambio di dominio ma dal fatto che il modello non è stato costruito per riconoscerle, e questo avrebbe falsato la misura.

Dopo l'esclusione delle due classi restano 14.394 immagini utilizzabili. Il conteggio tiene già conto di un file scartato perché non è un'immagine valida, un problema descritto più avanti nella sezione dedicata alla qualità dei dati.


#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    table.hline(),
    table.header(
      [Classe], [FER-2013 train], [FER-2013 test], [FANE],
    ),
    table.hline(),
    [angry],    [3.995],  [958],   [1.774],
    [disgust],  [436],    [111],   [1.378],
    [fear],     [4.097],  [1.024], [2.900],
    [happy],    [7.215],  [1.774], [1.921],
    [neutral],  [4.965],  [1.233], [1.835],
    [sad],      [4.830],  [1.247], [2.732],
    [surprise], [3.171],  [831],   [1.854],
    table.hline(),
    [*Totale*], [*28.709*], [*7.178*], [*14.394*],
    table.hline(),
  ),
  caption: [Numero di immagini per classe nei tre insiemi utilizzati.],
) <tab:conteggi-classi>
== Analisi esplorativa

Prima di addestrare i modelli è stata esaminata la composizione dei due dataset. La @fig:distribuzione riporta il numero di immagini per ciascuna classe nei tre insiemi.


#figure(
  image("img/class_distribution.png", width: 100%),
  caption: [
    Numero di immagini per classe in FER-2013 (addestramento e test) e in FANE.
  ],
) <fig:distribuzione>
FER-2013 risulta fortemente sbilanciato. La classe happy conta 7.215 immagini di addestramento mentre disgust ne ha soltanto 436, con un rapporto di circa sedici a uno fra la più numerosa e la meno numerosa. Il test set riproduce fedelmente le stesse proporzioni, il che è atteso dato che deriva dalla stessa raccolta. FANE è invece distribuito in modo più uniforme, senza classi altrettanto marginali.

Il confronto diretto fra le due composizioni è riportato nella @fig:bilanciamento, dove i conteggi sono espressi in percentuale sul totale di ciascun dataset. Mostra che i due dataset non differiscono solo per come sono fatte le immagini ma anche per quanto pesa ciascuna emozione al loro interno.


#figure(
  image("img/class_balance_comparison.png", width: 90%),
  caption: [
    Composizione percentuale di FER-2013 e FANE a confronto. Le proporzioni
    delle singole classi cambiano in modo sostanziale fra i due dataset.
  ],
) <fig:bilanciamento>

Le differenze sono marcate. La classe disgust rappresenta appena l'1,5 per cento di FER-2013 e sale al 9,6 per cento in FANE, quindi il suo peso aumenta di oltre sei volte. In direzione opposta la classe happy scende dal 25,1 al 13,3 per cento, e fear passa dal 14,3 al 20,2. Anche le altre classi si spostano, seppure in misura minore.

Questo ha una conseguenza diretta sulla scelta della metrica. L'accuracy conta semplicemente quante immagini sono state classificate correttamente, quindi le classi più numerose incidono di più sul risultato finale. Se la composizione cambia fra i due dataset, l'accuracy cambia in parte anche solo per questo motivo, indipendentemente da quanto il modello sia effettivamente peggiorato. Confrontare l'accuracy su FER-2013 con quella su FANE significherebbe quindi mescolare il calo di prestazioni con il cambio di composizione. È la ragione per cui i risultati vengono riportati anche in macro-F1, una metrica che assegna lo stesso peso a tutte le classi e che viene descritta nella sezione dedicata alle metriche.

== Confronto visivo fra i domini

Le differenze fra i due dataset non riguardano soltanto i numeri. La @fig:esempi mostra alcune immagini scelte a caso per ogni classe, con FER-2013 a sinistra e FANE a destra.


#figure(
  grid(
    columns: 2,
    gutter: 1em,
    image("img/samples_fer2013.png", width: 100%),
    image("img/samples_fane.png", width: 100%),
  ),
  caption: [
    Esempi casuali per ciascuna classe. A sinistra FER-2013, a destra FANE.
  ],
) <fig:esempi>

Le immagini di FER-2013 sono ritagli stretti e uniformi, in scala di grigi, con il volto sempre inquadrato allo stesso modo e con pochissimo sfondo. FANE è visibilmente più eterogeneo. Le fotografie sono a colori, la risoluzione varia da un'immagine all'altra, l'inquadratura non segue una regola fissa e insieme alle fotografie compaiono anche illustrazioni e fotomontaggi.

Guardare le due griglie affiancate rende evidente quanto siano distanti i due domini, e serve a dare un'idea concreta di cosa significhi domain shift prima ancora di misurarlo. Un modello addestrato sulle immagini di sinistra si troverà a lavorare su quelle di destra senza aver mai visto nulla di simile.

== Qualità dei dati

Durante l'analisi esplorativa sono emersi alcuni problemi nei dati, che vale la pena riportare perché hanno avuto conseguenze concrete sul codice e perché aiutano a interpretare i risultati.

Il caso più singolare riguarda FANE. Il file happy/happy1283.jpg non è un'immagine, ma un notebook Jupyter salvato con l'estensione sbagliata. Cancellarlo dalla copia locale non sarebbe stato sufficiente, dato che il dataset viene riscaricato a ogni esecuzione sull'ambiente di addestramento remoto, quindi il caricamento dei dati è stato modificato in modo da scartare automaticamente i file che non possono essere letti come immagini. È per questo motivo che le immagini utilizzabili di FANE sono 14.394 e non 14.395.

Anche FER-2013 presenta immagini che non dovrebbero esserci. Nella griglia di esempi della classe happy è comparso lo screenshot di un segnaposto per immagine mancante, contenente soltanto del testo e nessun volto. Diverse altre immagini riportano il marchio in trasparenza dei siti di fotografie da cui sono state prelevate, sovrapposto al viso. Sono conseguenze della raccolta automatica e sono note in letteratura, e contribuiscono a spiegare perché anche gli annotatori umani si fermino intorno al sessantacinque per cento.

FANE presenta invece un problema diverso, legato alla sua eterogeneità. Accanto alle fotografie si trovano illustrazioni e fotomontaggi, la qualità varia molto da un'immagine all'altra e alcune etichette appaiono discutibili, con volti che a un osservatore esterno sembrerebbero esprimere un'emozione differente da quella indicata. Su quest'ultimo punto va fatta una precisazione importante. L'eterogeneità di FANE non è stata trattata come rumore da ripulire, perché è parte integrante di ciò che il lavoro vuole misurare. Un sistema impiegato in una situazione reale incontra esattamente questo genere di immagini, quindi rimuovere i casi difficili avrebbe reso la valutazione più favorevole ma meno informativa.

== Preprocessing

I due modelli richiedono immagini in formati diversi, quindi la pipeline di caricamento prevede due rami distinti che vengono scelti in base al modello da addestrare.


#figure(
  table(
    columns: 4,
    align: (left, center, center, left),
    stroke: none,
    table.hline(),
    table.header(
      [Modello], [Canali], [Dimensione], [Normalizzazione],
    ),
    table.hline(),
    [CNN custom], [1 (grigio)], [48 × 48], [divisione per 255, intervallo $[0, 1]$],
    [Xception],   [3 (RGB)],    [71 × 71], [funzione del modello, intervallo $[-1, 1]$],
    table.hline(),
  ),
  caption: [I due rami di preprocessing.],
) <tab:preprocessing>

Il primo ramo mantiene le immagini nel formato originale di FER-2013 e si limita a riportare i valori dei pixel nell'intervallo fra zero e uno. Il secondo deve invece adattarsi ai requisiti di Xception, che accetta immagini a tre canali con lato minimo di 71 pixel e si aspetta valori normalizzati fra meno uno e uno tramite la funzione fornita insieme al modello. Le immagini di FER-2013 vengono quindi ingrandite e replicate sui tre canali. È importante notare che questo passaggio non aggiunge informazione, perché il contenuto resta quello di un'immagine 48 per 48 in scala di grigi, e il punto verrà ripreso nella discussione dei risultati.

All'insieme di addestramento viene applicata una serie di trasformazioni casuali, cioè il ribaltamento orizzontale, una rotazione fino a circa undici gradi e uno zoom e una traslazione fino al dieci per cento. Le trasformazioni vengono rigenerate a ogni epoca, quindi il modello non vede mai due volte esattamente la stessa immagine. Servono ad aumentare la varietà dei dati e soprattutto a rendere più difficile per la rete memorizzare le caratteristiche di acquisizione di FER-2013, che sono proprio il tipo di scorciatoia che si vuole evitare in un lavoro sulla generalizzazione. Gli insiemi di validazione e di test non vengono mai alterati.

Dall'insieme di addestramento di FER-2013 è stato infine ricavato un insieme di validazione pari al dieci per cento del totale, ottenendo 25.839 immagini per l'addestramento e 2.870 per la validazione. La suddivisione avviene con lo stesso seme casuale in entrambi i casi, in modo che i due sottoinsiemi risultino disgiunti e nessuna immagine finisca in tutti e due. L'insieme di validazione ha un ruolo preciso, perché è l'unico su cui vengono
prese decisioni durante l'addestramento. È lì che viene deciso quando fermarsi,
quale versione del modello conservare e, fra tutte le configurazioni provate,
quale venga scelta come modello finale. Il test set di FER-2013 e l'intero FANE non intervengono mai in questa fase e vengono usati soltanto alla fine, a modello ormai definito. Questa separazione è ciò che garantisce che i valori riportati nei risultati siano stime non distorte, perché nessuna scelta è stata fatta guardando i dati su cui il modello viene poi giudicato.

= Metodologia

== CNN residuale custom

Il primo modello è una rete convoluzionale progettata per questo lavoro e
addestrata da zero. La struttura è composta da tre blocchi residui in sequenza,
con un numero di filtri che raddoppia a ogni blocco passando da 64 a 128 e infine
a 256, seguiti da una testa di classificazione.

Ogni blocco contiene due convoluzioni 3 per 3, ciascuna seguita da una
normalizzazione batch e da una attivazione LeakyReLU. L'ingresso del blocco viene
poi sommato all'uscita di queste due convoluzioni attraverso una connessione
residua, quando serve adattando il numero di canali con una convoluzione 1 per 1.
Al termine del blocco una operazione di max pooling dimezza la risoluzione e uno
SpatialDropout spegne casualmente alcune mappe di caratteristiche.

Le connessioni residue sono la scelta più caratterizzante dell'architettura e
servono a rendere addestrabile una rete profonda a partire da pesi casuali.
Durante la backpropagation il gradiente deve attraversare tutti i livelli per
arrivare a quelli iniziali, e a ogni passaggio tende ad attenuarsi, al punto che
nei primi livelli diventa troppo debole per produrre aggiornamenti utili. La
connessione residua offre al gradiente un percorso diretto che salta le
convoluzioni, quindi i livelli iniziali continuano a ricevere un segnale di
apprendimento anche quando la rete è profonda. C'è anche un secondo effetto, cioè
che il blocco non deve imparare l'intera trasformazione ma soltanto una
correzione da sommare a quanto già ricevuto, il che rende l'ottimizzazione più
semplice.

Lo SpatialDropout cresce da 0,2 a 0,4 procedendo in profondità. A differenza del
dropout ordinario spegne intere mappe di caratteristiche invece di singoli
valori, e su dati convoluzionali questo è più efficace, perché i pixel vicini
sono fortemente correlati e disattivarli uno alla volta lascia comunque passare
l'informazione. La percentuale cresce perché i blocchi più profondi hanno molti
più parametri e tendono a specializzarsi di più. A questo si aggiunge una
regolarizzazione L2 applicata a tutti i pesi convoluzionali e densi.

#figure(
  table(
    columns: 4,
    align: (left, center, center, right),
    stroke: none,
    table.hline(),
    table.header(
      [Blocco], [Filtri], [Uscita], [Parametri],
    ),
    table.hline(),
    [Ingresso],          [--],  [48 × 48 × 1],   [--],
    [Blocco residuo 1],  [64],  [24 × 24 × 64],  [38.272],
    [Blocco residuo 2],  [128], [12 × 12 × 128], [230.912],
    [Blocco residuo 3],  [256], [6 × 6 × 256],   [920.576],
    [Testa],             [--],  [7],             [68.359],
    table.hline(),
    [*Totale*], [], [], [*1.258.119*],
    table.hline(),
  ),
  caption: [Struttura della CNN custom. Lo SpatialDropout vale 0,2 nel primo
    blocco, 0,3 nel secondo e 0,4 nel terzo.],
) <tab:architettura-cnn>

Il modello conta poco più di 1,2 milioni di parametri, una dimensione contenuta
scelta deliberatamente in rapporto alla quantità di dati disponibili.

== Xception e le strategie di fine-tuning

Il secondo modello parte da Xception pre-addestrata su ImageNet, presa senza i
livelli finali di classificazione. Su una rete di questo tipo si può decidere
quanta parte riaddestrare, e la scelta non è binaria. All'estremo più
conservativo si riaddestra solo la testa di classificazione, lasciando la rete
pre-addestrata a fare da estrattore di caratteristiche fisso. All'estremo opposto
si riaddestra tutto. In mezzo si può scongelare a partire da un livello
qualsiasi, e più ci si sposta verso l'ingresso più parametri diventano liberi di
adattarsi.

#figure(
  table(
    columns: 4,
    align: (left, left, right, right),
    stroke: none,
    table.hline(),
    table.header(
      [Strategia], [Primo livello scongelato], [Parametri addestrabili], [Quota],
    ),
    table.hline(),
    [Solo testa],       [--],                 [526.599],    [2,5 %],
    [Ultimo blocco],    [`block14_sepconv1`], [5.268.231],  [24,6 %],
    [Rete intera],      [tutti],              [21.279.023], [99,5 %],
    table.hline(),
  ),
  caption: [Le tre profondità di fine-tuning. La prima è usata solo come fase
    iniziale di warm-up e non come configurazione a sé.],
) <tab:sblocco>

L'argomento a favore dello scongelamento parziale è il rapporto fra parametri e
dati. I livelli iniziali di una rete convoluzionale apprendono caratteristiche
molto generali come bordi, texture e transizioni di intensità, che restano valide
anche cambiando dominio, mentre i livelli finali codificano informazione
specifica del compito su cui la rete è stata addestrata. Sono questi ultimi a
dover essere riadattati. Scongelare anche i primi significa riscrivere pesi
appresi su oltre un milione di immagini usando le 25.839 a disposizione, con il
rischio di rovinarli invece di migliorarli.

È un argomento ragionevole, ma resta un'ipotesi finché non viene verificata. Per
questo motivo la profondità non è stata fissata a priori e sono state confrontate
due configurazioni, lo scongelamento del solo ultimo blocco convolutivo e quello
dell'intera rete.

Una seconda scelta riguarda i livelli di normalizzazione batch della rete
pre-addestrata, che sono stati mantenuti congelati in tutte le configurazioni,
anche all'interno della porzione scongelata. Questi livelli conservano al loro
interno delle statistiche calcolate durante l'addestramento su ImageNet e le
usano per normalizzare gli ingressi. Lasciandoli liberi, quelle statistiche
verrebbero ricalcolate sui nostri lotti da 32 immagini, che sono pochi e
provengono da una distribuzione molto diversa. Il risultato sarebbe uno
spostamento delle normalizzazioni a cui i pesi successivi non sono più adatti,
con un peggioramento proprio delle caratteristiche che il transfer learning
dovrebbe preservare. Tenerli congelati è la precauzione standard in questi casi.

== Classification head condivisa

I due modelli terminano con la stessa identica testa di classificazione,
composta da un global average pooling, un livello denso da 256 unità, una
normalizzazione batch, una attivazione LeakyReLU, un dropout al 50 per cento e
infine il livello di uscita a 7 unità con attivazione softmax.

La scelta non è casuale. Se i due modelli avessero classificatori diversi, una
differenza nei risultati potrebbe dipendere tanto dall'estrattore di
caratteristiche quanto dal classificatore, e non ci sarebbe modo di distinguere i
due contributi. Tenendo la testa identica, l'unica cosa che cambia fra i due
modelli è il modo in cui vengono estratte le caratteristiche, quindi è a quello
che si può attribuire la differenza osservata.

== Il piano sperimentale

Sono stati addestrati cinque modelli, riportati nella @tab:piano.

#figure(
  table(
    columns: 5,
    align: (left, left, center, center, left),
    stroke: none,
    table.hline(),
    table.header(
      [Run], [Modello], [Batch], [Warm-up], [Parte riaddestrata],
    ),
    table.hline(),
    [`custom_cnn_bs32`],      [CNN custom], [32], [--],       [tutta, da zero],
    [`custom_cnn_bs64`],      [CNN custom], [64], [--],       [tutta, da zero],
    [`xception_1phase`],      [Xception],   [32], [no],       [ultimo blocco, 24,6 %],
    [`xception_2phase`],      [Xception],   [32], [5 epoche], [ultimo blocco, 24,6 %],
    [`xception_full`],        [Xception],   [32], [5 epoche], [rete intera, 99,5 %],
    table.hline(),
  ),
  caption: [Le cinque configurazioni addestrate.],
) <tab:piano>

Ogni coppia isola una singola variabile. Le due run della CNN custom
differiscono solo per il batch size. La prima e la seconda run di Xception
differiscono solo per la presenza del warm-up. La seconda e la terza differiscono
solo per la profondità dello scongelamento, dato che condividono warm-up, batch
size e tasso di apprendimento. In questo modo ogni confronto ha una sola causa
possibile.

== Protocollo di addestramento

Tutti i modelli sono stati addestrati con l'ottimizzatore Adam e la funzione di
perdita cross-entropy categorica, con un tetto massimo di 100 epoche.

La CNN custom è stata addestrata sia con lotti da 32 sia da 64, per misurare
l'effetto del batch size su una rete addestrata da zero. Xception è stata
addestrata solo con lotti da 32, che è la dimensione standard indicata per questa rete.

Il tasso di apprendimento è fissato a 1e-3 per la rete addestrata da zero e a
1e-4 per il fine-tuning. La rete che parte da pesi casuali ha bisogno di passi
ampi, perché deve percorrere molta strada prima di arrivare a una configurazione
sensata. Il modello pre-addestrato si trova invece già in una buona regione dello
spazio dei parametri, e un passo dieci volte più grande rischierebbe di
allontanarlo nelle prime iterazioni, cancellando l'informazione appresa su
ImageNet che è proprio il motivo per cui lo si è scelto. Lo stesso valore di 1e-4
è stato usato sia per lo scongelamento parziale sia per quello completo, così che
fra le due configurazioni cambi soltanto il numero di livelli addestrabili.

Il warm-up risponde a un problema specifico del fine-tuning. Nelle prime
iterazioni la testa di classificazione è ancora inizializzata a caso e produce
gradienti molto grandi, che si propagano all'indietro fino ai pesi pre-addestrati
e rischiano di guastarli prima che la testa abbia imparato qualcosa di utile. Per
evitarlo si tiene l'intera rete pre-addestrata congelata per le prime cinque
epoche, addestrando solo la testa, e solo dopo si scongela la porzione prevista e
si prosegue al tasso ridotto. La run senza warm-up serve a verificare se questa
precauzione produca davvero una differenza.

L'arresto dell'addestramento è affidato a un meccanismo di early stopping che
sorveglia la perdita sull'insieme di validazione e interrompe quando questa non
migliora per dieci epoche consecutive, ripristinando poi i pesi dell'epoca
migliore. In parallelo un meccanismo di checkpoint salva su disco la versione del
modello con la perdita di validazione più bassa. Il tetto di 100 epoche va quindi
inteso come limite superiore e non come durata effettiva, perché è l'early
stopping a decidere quando fermarsi.

Tutte le sorgenti di casualità, cioè l'inizializzazione dei pesi, il mescolamento
dei dati, le trasformazioni di augmentation e la suddivisione fra addestramento e
validazione, sono governate da un unico seme fissato a 42, in modo da rendere gli
esperimenti ripetibili.

== Metriche

I risultati sono espressi con tre metriche affiancate.

La prima è l'accuracy, cioè la percentuale di immagini classificate correttamente
sul totale. È la misura più immediata da leggere e permette il confronto con i
valori di riferimento riportati nello stato dell'arte, ma ha un difetto
importante quando le classi non sono bilanciate, perché le classi più numerose
incidono di più sul risultato e una classe rara può essere ignorata quasi del
tutto senza che il numero finale ne risenta.

Le altre due nascono dall'F1, definita come media armonica fra precision e
recall.

$ F_1 = 2 dot (P dot R) / (P + R) $

L'F1 è però una misura binaria, definita per una classe contro tutte le altre.
Con sette classi si ottengono sette valori distinti, e per arrivare a un unico
numero occorre decidere come mediarli. Le tre convenzioni possibili portano a
risultati diversi.

La media micro calcola i conteggi complessivamente su tutte le classi e nel caso
di classificazione a etichetta singola coincide esattamente con l'accuracy,
quindi non aggiunge nulla a quanto già riportato e non viene usata.

La media pesata assegna a ciascuna classe un peso proporzionale al numero di
immagini che contiene. Descrive bene il comportamento del modello su un singolo
dataset, perché tiene conto di come quel dataset è davvero composto, ma non è
adatta al confronto fra i due, dato che i pesi cambierebbero passando da FER-2013
a FANE e la differenza fra i due valori mescolerebbe il calo di prestazioni con
il diverso bilanciamento. Viene quindi riportata per ciascun dataset separatamente
e non viene usata per misurare il divario.

La media macro calcola invece la media semplice delle sette F1, assegnando a ogni
classe lo stesso peso indipendentemente da quante immagini contenga.

$ "macro-F"_1 = 1/C sum_(c=1)^C F_(1,c) $

Essendo la ponderazione identica sui due dataset, la differenza fra la macro-F1
su FER-2013 e quella su FANE misura soltanto il peggioramento del modello. È per
questo motivo che il divario di generalizzazione viene quantificato in macro-F1,
mentre l'accuracy resta riportata a fianco per permettere il confronto con la
letteratura.

== Sbilanciamento delle classi

Una domanda naturale a questo punto è perché lo sbilanciamento sia stato
affrontato scegliendo la metrica invece che correggendo i dati. Le tecniche più
comuni sono il sottocampionamento delle classi più numerose e il
sovracampionamento di quelle rare, ma nessuna delle due è sembrata adatta.

Il sottocampionamento avrebbe portato tutte le classi al livello della più rara,
cioè disgust con 436 immagini, lasciando 3.052 immagini di addestramento contro
le 28.709 disponibili. Significa rinunciare a quasi il novanta per cento dei dati
per una rete che deve imparare da zero, un prezzo troppo alto. Il
sovracampionamento non butta via nulla, ma si limita a ripetere le stesse 436
immagini di disgust molte volte, quindi aggiunge occasioni di memorizzarle senza
aggiungere varietà.

Inoltre, lo sbilanciamento di
FER-2013 non è un difetto da correggere prima di misurare, ma è parte di ciò che
il lavoro vuole misurare. Un modello addestrato su un dataset in cui disgust vale
l'uno e mezzo per cento eredita quella proporzione e se la porta dietro anche
quando viene applicato a un dataset in cui la stessa classe è oltre sei volte più
frequente. Riequilibrare i dati avrebbe nascosto proprio il fenomeno che i
risultati mettono in evidenza. Resta il fatto che una terza strada, il class
weighting, non è stata provata, e il punto viene ripreso fra i limiti del lavoro.

Resta una terza strada, il class weighting, che pesa le classi nella funzione di
perdita senza toccare i dati. Non è stata adottata perché attenuerebbe il prior che il modello eredita dal dataset
sorgente, che è uno dei fenomeni che il lavoro vuole osservare.


= Risultati

== Andamento dell'addestramento

Nessuna delle cinque configurazioni ha raggiunto il tetto di 100 epoche, perché
l'early stopping è sempre intervenuto prima. È un dettaglio importante, perché
significa che i modelli si sono fermati per convergenza e non perché il budget
fosse esaurito. Un addestramento troncato lascerebbe il dubbio che con più tempo
il risultato sarebbe migliorato, dubbio che in questo caso non si pone.

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header(
      [Run], [Epoche], [Epoca migliore], [val\_loss], [val\_accuracy], [Scarto train − val],
    ),
    table.hline(),
    [`custom_cnn_bs32`], [27], [17], [1,2863], [0,5617], [−0,0389],
    [`custom_cnn_bs64`], [57], [47], [1,1665], [0,6066], [−0,0060],
    [`xception_1phase`], [31], [21], [1,3004], [0,5422], [+0,0267],
    [`xception_2phase`], [31], [21], [1,3034], [0,5362], [+0,0335],
    [`xception_full`],   [21], [11], [0,9553], [0,6592], [+0,0222],
    table.hline(),
  ),
  caption: [Riepilogo dell'addestramento delle cinque configurazioni.],
) <tab:addestramento>

#figure(
  image("img/training_curves.png", width: 100%),
  caption: [
    Loss e accuracy per epoca dei due modelli selezionati. La linea tratteggiata
    indica l'insieme di addestramento, quella continua l'insieme di validazione.
    Il pallino segna l'epoca ripristinata dall'early stopping e la linea
    punteggiata verticale la fine del warm-up.
  ],
) <fig:curve>

Tutte le configurazioni di Xception partono da una accuracy di validazione
intorno al 40 per cento già alla prima epoca, contro il 24 della rete addestrata
da zero. Le caratteristiche apprese su ImageNet forniscono quindi un punto di
partenza chiaramente migliore. Ciò che cambia è se quel vantaggio venga poi
mantenuto, e la risposta dipende da quanta rete viene lasciata libera di
adattarsi.

Il segno dello scarto fra addestramento e validazione separa nettamente i due
tipi di modello. Per tutte le configurazioni di Xception è positivo, quindi il
modello va meglio sui dati che ha già visto. Per la CNN custom è negativo, cioè
la validazione va leggermente meglio dell'addestramento. Non è un errore, dipende
dal fatto che le metriche di addestramento sono calcolate sulle immagini
trasformate dall'augmentation, che sono più difficili di quelle originali usate
in validazione.

== Confronto fra le configurazioni

La @tab:risultati riporta i risultati di tutte e cinque le run sui due test set.

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header(
      [Run],
      [Acc. FER], [Macro-F1], [F1 pes.],
      [Acc. FANE], [Macro-F1], [F1 pes.],
    ),
    table.hline(),
    [`custom_cnn_bs32`], [55,91 %], [0,4481], [0,5372], [34,83 %], [0,2926], [0,2905],
    [`custom_cnn_bs64`], [61,12 %], [0,5203], [0,5930], [34,62 %], [0,2941], [0,2883],
    [`xception_1phase`], [53,39 %], [0,4824], [0,5191], [28,55 %], [0,2493], [0,2470],
    [`xception_2phase`], [52,80 %], [0,4697], [0,5239], [29,51 %], [0,2488], [0,2561],
    [`xception_full`],   [65,85 %], [0,6210], [0,6556], [37,62 %], [0,3393], [0,3355],
    table.hline(),
  ),
  caption: [Risultati delle cinque configurazioni sui due dataset.],
) <tab:risultati>

Il primo confronto riguarda il warm-up, e la risposta è che sembra non servire. Le due run
`xception_1phase` e `xception_2phase` differiscono soltanto per la presenza della
fase iniziale a base congelata, e finiscono praticamente sovrapposte. Entrambe si
fermano alla stessa epoca con la stessa epoca migliore, la val\_loss differisce di
tre millesimi e la macro-F1 su FER-2013 di poco più di un punto, a favore della
versione senza warm-up. La precauzione contro i gradienti della testa casuale non
produce alcun effetto misurabile in questo contesto.

Il secondo confronto riguarda la profondità dello scongelamento, ed è il
risultato più netto del lavoro.

#figure(
  image("img/finetune_strategy.png", width: 90%),
  caption: [
    Macro-F1 delle tre strategie di fine-tuning sui due dataset, a parità di
    batch size. Sotto ogni gruppo è indicata la quota di parametri addestrabili.
  ],
) <fig:strategie>

#figure(
  image("img/xception_strategy_curves.png", width: 100%),
  caption: [
    Curve di validazione delle tre strategie. Lo scongelamento completo raggiunge
    una loss molto più bassa e in molte meno epoche.
  ],
) <fig:curve-strategie>

Passando dal 24,6 al 99,5 per cento di parametri addestrabili l'accuracy su
FER-2013 sale dal 52,80 al 65,85 per cento e la macro-F1 da 0,4697 a 0,6210. Il
guadagno si ripete su FANE, dove l'accuracy passa dal 29,51 al 37,62 per cento.
La val\_loss scende da 1,3034 a 0,9553, quindi il vantaggio è visibile già in
validazione e non è un effetto emerso solo in fase di test.

Il risultato smentisce l'argomento presentato nella metodologia. Riaddestrare
l'intera rete su 25.839 immagini non ha rovinato i pesi appresi su ImageNet, li ha
resi molto più utili. Il fine-tuning parziale, quello che l'argomento sul rapporto
fra parametri e dati avrebbe dovuto favorire, è invece la configurazione peggiore
delle tre e resta sotto perfino alla rete addestrata da zero.

Il terzo confronto riguarda il batch size sulla CNN custom.

#figure(
  image("img/batch_size_comparison.png", width: 80%),
  caption: [
    Macro-F1 della CNN custom addestrata con lotti da 32 e da 64, sui due
    dataset.
  ],
) <fig:batch>

Con lotti da 64 la rete ottiene 61,12 per cento di accuracy su FER-2013 contro
55,91, cioè 5,21 punti in più, e la macro-F1 sale da 0,4481 a 0,5203. Su FANE la
differenza sparisce, con 34,62 contro 34,83 per cento. Il batch size più grande
aiuta quindi in dominio senza cambiare nulla fuori dominio.

Va notato che la run a batch 32 si è fermata dopo 27 epoche contro le 57 dell'altra,
ma il confronto in epoche è ingannevole. Con lotti più piccoli ogni epoca contiene
il doppio degli aggiornamenti dei pesi, quindi le due run hanno ricevuto una
quantità di addestramento più simile di quanto i numeri suggeriscano.

Sulla base della sola val\_loss sono state selezionate `custom_cnn_bs64` e
`xception_full` come rappresentanti delle due architetture. Tutte le analisi che
seguono si riferiscono a queste due.

== Prestazioni in dominio

Xception con fine-tuning completo raggiunge il 65,85 per cento di accuracy su
FER-2013, un valore che va letto insieme ai riferimenti dello stato dell'arte. Si
colloca sostanzialmente sulla prestazione stimata degli annotatori umani e circa
cinque punti sotto i modelli migliori pubblicati in letteratura, un risultato considerabile come buono.

La CNN custom si ferma al 61,12 per cento. La differenza vale 4,74 punti
percentuali con un intervallo di confidenza al 95 per cento di ±1,57, quindi al di
fuori di ciò che si potrebbe attribuire al caso. Il transfer learning risulta il
modello più accurato dei due, ma soltanto nella sua forma completa.

#figure(
  grid(
    columns: 2,
    gutter: 0.5em,
    image("img/custom_cnn_fer2013_confusion.png", width: 100%),
    image("img/xception_fer2013_confusion.png", width: 100%),
  ),
  caption: [
    Matrici di confusione su FER-2013, normalizzate per riga. A sinistra la CNN
    custom, a destra Xception.
  ],
) <fig:confusione-fer>

Le matrici della @fig:confusione-fer hanno una diagonale ben visibile in entrambi
i casi. Le classi happy e surprise sono quelle riconosciute meglio. La differenza
più interessante riguarda disgust, la classe più rara, dove la CNN custom si ferma
a un recall di 0,099 mentre Xception arriva a 0,414. Il modello pre-addestrato
riesce quindi a imparare una classe che nel training set vale l'uno e mezzo per
cento, cosa che la rete addestrata da zero non fa.

== Prestazioni fuori dominio

Gli stessi due modelli sono stati poi valutati su FANE, senza alcun riadattamento.
Xception scende al 37,62 per cento di accuracy e la CNN custom al 34,62. La
differenza vale 3,00 punti con un intervallo di ±1,11, quindi il vantaggio di
Xception si conserva anche fuori dominio, seppure ridotto.

#figure(
  grid(
    columns: 2,
    gutter: 0.5em,
    image("img/custom_cnn_fane_confusion.png", width: 100%),
    image("img/xception_fane_confusion.png", width: 100%),
  ),
  caption: [
    Matrici di confusione su FANE, normalizzate per riga. A sinistra la CNN
    custom, a destra Xception.
  ],
) <fig:confusione-fane>

Le matrici della @fig:confusione-fane sono la parte più istruttiva dell'intera
sezione, perché mostrano che il peggioramento non è una degradazione uniforme ma
ha una struttura molto precisa.

La colonna corrispondente a disgust è quasi vuota in entrambe le matrici. Il recall
vale 0,021 per la CNN custom e 0,078 per Xception, quindi su 1.378 immagini di
disgust presenti in FANE la prima ne riconosce meno di trenta. Il dato
interessante è che la precision resta alta, pari a 0,592 e 0,507, quindi quando un
modello si decide a rispondere disgust ha spesso ragione. Semplicemente non lo fa
quasi mai, perché ha imparato da FER-2013 che si tratta di una classe rarissima e
si porta dietro quella convinzione anche dove non vale più. Vale la pena notare
che questo accade anche a Xception, che in dominio quella classe l'aveva imparata
bene.

All'estremo opposto le predizioni si concentrano su happy e neutral. La CNN custom
raggiunge su FANE un recall di 0,872 per happy e di 0,784 per neutral, valori che a
prima vista sembrerebbero ottimi, ma la precision corrispondente crolla a 0,445 e
0,274. I modelli stanno cioè rispondendo happy o neutral molto più spesso di
quanto dovrebbero, assegnando a queste due classi anche immagini che appartengono
ad altre. Il caso più evidente riguarda surprise, di cui il 38 per cento delle
immagini viene classificato come happy, probabilmente perché una bocca aperta
risulta ambigua una volta che cambiano illuminazione, inquadratura e risoluzione.
Anche sad si comporta male, con il 54 per cento delle immagini assegnato a neutral
contro un recall proprio di appena 0,17.

Il comportamento è sostanzialmente identico nei due modelli, il che è già un primo
indizio del fatto che il problema non dipenda dall'architettura scelta.

== Divario di generalizzazione

Mettendo a confronto le prestazioni dentro e fuori dominio si ottiene il risultato
centrale del lavoro.

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header(
      [Modello], [Accuracy FER], [Accuracy FANE], [Divario], [IC 95 %],
    ),
    table.hline(),
    [CNN custom], [61,12 %], [34,62 %], [26,50 pp], [± 1,37],
    [Xception],   [65,85 %], [37,62 %], [28,23 pp], [± 1,35],
    table.hline(),
    [*Differenza*], [], [], [*1,74 pp*], [*± 1,93*],
    table.hline(),
  ),
  caption: [Divario di generalizzazione in accuracy. La differenza fra i due
    divari non è statisticamente significativa.],
) <tab:divario>

Entrambi i modelli perdono più di venticinque punti percentuali di accuracy nel
passaggio da FER-2013 a FANE. La CNN custom scende di 26,50 punti e Xception di
28,23, e la differenza fra i due valori vale 1,74 punti con un intervallo di
confidenza di ±1,93. Poiché l'intervallo include lo zero, la differenza non è
statisticamente significativa e non si può affermare che uno dei due modelli
generalizzi meglio dell'altro.

#figure(
  image("img/generalization_gap_slope.png", width: 80%),
  caption: [
    Caduta della macro-F1 nel passaggio da FER-2013 a FANE.
  ],
) <fig:pendenza>

Lo stesso confronto in macro-F1 racconta la stessa cosa. La CNN custom perde 22,62
punti, pari al 43,5 per cento del valore di partenza, e Xception ne perde 28,17,
pari al 45,4 per cento. In termini relativi le due perdite sono quindi molto
vicine, e Xception paga in valore assoluto una caduta maggiore soltanto perché
parte da più in alto.

Il pre-addestramento su ImageNet ha quindi migliorato le prestazioni in valore
assoluto su entrambi i dataset, ma non ha ridotto la frazione che si perde
attraversando il cambio di dominio.

== Analisi per classe

L'ultimo livello di dettaglio riguarda il comportamento sulle singole classi.

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header(
      [Classe], [CNN / FER], [CNN / FANE], [Xcep / FER], [Xcep / FANE],
    ),
    table.hline(),
    [angry],    [0,528], [0,321], [0,564], [0,320],
    [disgust],  [0,169], [0,041], [0,500], [0,135],
    [fear],     [0,295], [0,148], [0,473], [0,221],
    [happy],    [0,847], [0,590], [0,863], [0,602],
    [neutral],  [0,584], [0,407], [0,635], [0,425],
    [sad],      [0,465], [0,218], [0,533], [0,286],
    [surprise], [0,753], [0,336], [0,779], [0,386],
    table.hline(),
  ),
  caption: [F1 per classe nelle quattro combinazioni di modello e dataset.],
) <tab:f1-classi>

La cosa più evidente è che i due modelli ordinano le classi quasi allo stesso
modo. In tutte e quattro le colonne happy è la classe con l'F1 più alta e disgust
quella con l'F1 più bassa, con surprise stabilmente al secondo posto. La
correlazione fra i recall per classe dei due modelli su FANE vale 0,989, quindi i
due sbagliano sostanzialmente sulle stesse immagini.

Il calo verso FANE non colpisce però tutte le classi nella stessa misura. Happy e
neutral perdono circa il trenta per cento del proprio valore, surprise e fear
poco più della metà, mentre disgust ne perde tre quarti. Le classi già deboli in
dominio sono quindi anche quelle che crollano di più fuori dominio, e il risultato
è un modello che su FANE si riduce di fatto a distinguere fra poche categorie
frequenti.

Un dettaglio merita attenzione. Su FER-2013 il fine-tuning completo tripla l'F1
su disgust rispetto alla rete addestrata da zero, passando da 0,169 a 0,500,
quindi risolve in buona parte il problema della classe rara. Su FANE però quel
vantaggio si riduce a 0,135 contro 0,041, e la classe resta di fatto inutilizzabile
per entrambi. Imparare meglio una classe rara nel dominio di origine non basta a
trasferirla altrove.

== Costo computazionale

L'addestramento complessivo delle cinque configurazioni ha richiesto circa un'ora
e mezza di GPU.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    table.hline(),
    table.header(
      [Run], [Epoche], [Tempo totale], [Secondi per epoca],
    ),
    table.hline(),
    [`custom_cnn_bs32`], [27], [8 m 53 s],  [20],
    [`custom_cnn_bs64`], [57], [16 m 18 s], [17],
    [`xception_1phase`], [31], [22 m 41 s], [44],
    [`xception_2phase`], [31], [21 m 48 s], [42],
    [`xception_full`],   [21], [21 m 41 s], [62],
    table.hline(),
  ),
  caption: [Tempi di addestramento misurati su GPU.],
) <tab:tempi>

Il costo per epoca cresce da circa venti secondi per la CNN custom a sessantadue
per il fine-tuning completo, cioè poco più del triplo, il che è coerente con la
differenza di dimensione fra i due modelli. Il tempo totale però non segue la
stessa proporzione, perché lo scongelamento completo converge in molte meno epoche
e si ferma dopo ventuno. Il modello più costoso per epoca finisce quindi per
richiedere all'incirca lo stesso tempo complessivo delle altre due configurazioni
di Xception, pur ottenendo risultati molto migliori.


= Discussione

== Quanto conta riaddestrare

Il risultato più netto del lavoro è che il transfer learning vince o perde a
seconda di quanta rete si lascia libera di adattarsi. Con il solo ultimo blocco
convolutivo scongelato, cioè il 24,6 per cento dei parametri, Xception si ferma
al 52,80 per cento di accuracy e resta otto punti sotto la rete addestrata da
zero. Scongelando l'intera rete arriva al 65,85 e la supera di quasi cinque
punti. La stessa architettura, gli stessi pesi di partenza e lo stesso tasso di
apprendimento producono il modello peggiore o il migliore a seconda di questa
sola scelta.

Nella metodologia l'argomento a favore dello scongelamento parziale era il
rapporto fra parametri e dati, cioè il timore di rovinare pesi appresi su oltre
un milione di immagini riaddestrandoli su 25.839. I risultati lo smentiscono.
Riaddestrare tutto non ha distrutto le caratteristiche pre-addestrate, le ha rese
utilizzabili.

La spiegazione più coerente con quanto osservato è che le caratteristiche di
ImageNet siano semplicemente lontane dal compito. ImageNet contiene oggetti e
scene, e ciò che serve a distinguere un cane da un'automobile non è ciò che serve
a distinguere la paura dalla sorpresa, che dipende da differenze piccole e
localizzate nella configurazione di occhi, sopracciglia e bocca. Se le
caratteristiche sono distanti, congelarne tre quarti significa costringere il
modello a costruire la decisione su una rappresentazione inadatta, e il quarto
rimasto libero non basta a compensare.

Questo non significa che il pre-addestramento sia inutile, anzi. Tutte le
configurazioni di Xception partono da una accuracy di validazione intorno al 40
per cento già alla prima epoca, contro il 24 della rete addestrata da zero, e lo
scongelamento completo raggiunge il suo punto migliore all'undicesima epoca
mentre la CNN custom ne impiega quarantasette. I pesi di ImageNet restano quindi
un ottimo punto di partenza, purché siano liberi di muoversi.

Resta un fattore che nessuna delle configurazioni può aggirare, cioè la
risoluzione. Le immagini di FER-2013 sono 48 per 48 pixel in scala di grigi,
mentre Xception è stata addestrata su immagini a colori di lato 299. Per usarla
abbiamo dovuto ingrandirle e replicarle sui tre canali, operazione che non
aggiunge alcuna informazione. Buona parte della gerarchia convoluzionale è
costruita per riconoscere dettagli fini di texture e colore che nelle nostre
immagini non ci sono. È probabile che sia questo a spiegare perché il modello si
fermi intorno al valore degli annotatori umani invece di avvicinarsi ai risultati
migliori della letteratura, ma è un'ipotesi che questo lavoro non verifica.

== Il divario non dipende dall'architettura

Il risultato centrale riguarda però il comportamento dei due modelli quando cambia
il dominio.

Se il pre-addestramento su ImageNet conferisse una qualche robustezza al domain
shift, ci si dovrebbe aspettare che Xception perda meno nel passaggio a FANE, dal
momento che le sue caratteristiche sono state apprese su un insieme di immagini
enormemente più vario di FER-2013. Non è quello che accade. In accuracy la rete
addestrata da zero perde 26,50 punti e Xception 28,23, con una differenza di 1,74
punti e un intervallo di confidenza di ±1,93 che include lo zero, quindi non
distinguibile dal rumore.

Si potrebbe obiettare che il confronto in termini assoluti sfavorisca il modello
che parte da più in alto, e che andrebbe misurato quanto ciascun modello perde in
proporzione a quello che aveva. L'obiezione è ragionevole, ma il conto non cambia
la conclusione. In macro-F1 la rete addestrata da zero perde il 43,5 per cento del
proprio valore e Xception il 45,4 per cento, quindi con questa misura il modello
pre-addestrato risulta semmai leggermente meno robusto. Con nessuna delle due
letture il pre-addestramento si dimostra vantaggioso.

Il quadro complessivo è quindi che il pre-addestramento alza il livello assoluto
su entrambi i dataset, ma la frazione che si perde attraversando il cambio di
dominio resta la stessa. Sposta la curva verso l'alto senza renderla più piatta.

Da qui la tesi principale del lavoro. Il calo osservato non sembra una proprietà
dell'architettura né dell'inizializzazione dei pesi, ma delle differenze fra i due
dataset. I modelli non falliscono perché sono costruiti male, falliscono perché
le regolarità che hanno appreso da FER-2013 non valgono in FANE, e questo accade
indipendentemente da dove siano partiti. Che i due modelli sbaglino sulle stesse
classi, con una correlazione di 0,989 fra i rispettivi recall su FANE, è la prova
più diretta di questa lettura.

== Overfitting di dominio

C'è un aspetto di questi risultati che merita attenzione particolare, perché
riguarda il modo in cui l'overfitting viene normalmente diagnosticato.

Il controllo abituale consiste nel confrontare le prestazioni sul training set con
quelle sul validation set. Se il modello va molto meglio sui dati che ha già
visto, si conclude che stia memorizzando invece di generalizzare. Applicando
questo criterio alla CNN custom la risposta è netta, perché lo scarto fra training
e validation è rimasto negativo per tutto l'addestramento, cioè la validation
andava addirittura leggermente meglio. Secondo la diagnostica standard il modello
non stava overfittando affatto.

Eppure lo stesso modello perde ventisei punti di accuracy appena cambia dataset.
Si stava quindi specializzando su qualcosa, e quel qualcosa non era il singolo
esempio memorizzato ma le caratteristiche generali di FER-2013, cioè il tipo di
ritaglio, la scala di grigi, l'illuminazione e il modo in cui le etichette sono
state assegnate.

Il motivo per cui il controllo train contro validation non se ne accorge è
semplice. I due insiemi provengono dalla stessa raccolta, quindi condividono
esattamente quelle caratteristiche. Un modello che impara a sfruttarle ottiene
buoni risultati su entrambi, e lo scarto fra i due resta piccolo proprio mentre il
problema si sta creando.

La conseguenza pratica è che per rilevare questo tipo di sovradattamento il solo
split fra training e validation non basta, e serve un secondo dataset raccolto in
modo indipendente. È esattamente il ruolo che FANE ha avuto in questo lavoro, e la
sua utilità principale non è stata tanto fornire un numero aggiuntivo quanto
rendere visibile un problema che altrimenti sarebbe passato inosservato.

== Sbilanciamento e priori

L'analisi per classe mostra un fenomeno che collega i risultati alla composizione
dei dataset descritta nell'analisi esplorativa.

Il caso più chiaro è disgust. Su FANE la CNN custom ha un recall di 0,021 e
Xception di 0,078, quindi riconoscono rispettivamente il due e l'otto per cento
delle immagini di quella classe. La precision resta però alta, 0,592 e 0,507. Le
due cose insieme dicono una cosa precisa, cioè che i modelli quasi non predicono
mai disgust, ma quando lo fanno hanno spesso ragione. Il problema non è che non
sappiano riconoscere il disgusto, è che non se lo aspettano.

La spiegazione sta nel prior appreso durante il training. In FER-2013 disgust
rappresenta l'1,5 per cento delle immagini ed è la classe di gran lunga più rara.
Un classificatore addestrato su quella distribuzione impara due cose insieme, cioè
come è fatta l'espressione di disgusto e quanto è raro incontrarla, e usa entrambe
quando deve decidere. La seconda informazione è però legata al dataset di
provenienza e non si trasferisce, tanto che in FANE la stessa classe vale il 9,6
per cento, oltre sei volte tanto.

Un dettaglio rende il fenomeno ancora più evidente. Il fine-tuning completo su
FER-2013 impara disgust molto meglio della rete addestrata da zero, con una F1 di
0,500 contro 0,169 e un recall di 0,414 contro 0,099. Il problema della classe
rara in dominio è quindi in buona parte risolto. Su FANE però le due F1 diventano
0,135 e 0,041, e la classe resta inutilizzabile per entrambi. Imparare bene una
classe rara non basta, perché a non trasferirsi non è la conoscenza
dell'espressione ma l'aspettativa su quanto sia frequente.

La differenza di composizione fra i due dataset non è quindi soltanto un problema
di misura, che si risolve scegliendo la macro-F1. È anche un problema di modello,
perché il classificatore assorbe le proporzioni del training set e se le porta
dietro quando cambia dominio.

== Cosa dicono le tre metriche

Riportare accuracy, macro-F1 e F1 pesata affiancate non è una ridondanza, perché
la distanza fra le ultime due è a sua volta informativa.

La F1 pesata tiene conto di quanto ciascuna classe sia frequente nel dataset su
cui si valuta, mentre la macro-F1 le tratta tutte allo stesso modo. Quando la
pesata è molto più alta della macro significa che il modello va bene sulle classi
frequenti e male su quelle rare, cioè che sta appoggiandosi alla distribuzione più
che all'espressione. Su FER-2013 la CNN custom ha 0,5930 di pesata contro 0,5203 di
macro, una distanza di sette punti. Per il fine-tuning completo la stessa distanza
si dimezza, 0,6556 contro 0,6210. Il modello pre-addestrato è quindi non solo più
accurato ma anche più equilibrato fra le classi.

Su FANE il rapporto si rovescia e la pesata scende sotto la macro per entrambi i
modelli, 0,2883 contro 0,2941 e 0,3355 contro 0,3393. Il motivo è che in FANE le
classi più frequenti sono fear e sad, che sono proprio quelle su cui i modelli
vanno peggio. La stessa metrica che in dominio premiava i modelli, fuori dominio
li penalizza, e questo mostra concretamente perché il divario vada misurato in
macro-F1 e non con una metrica pesata sulla composizione.


= Limiti

I risultati vanno letti tenendo conto di alcuni limiti.

*Un solo addestramento per configurazione*. Ogni modello è stato addestrato una
volta sola, con il seed fissato a 42, quindi la variabilità dovuta
all'inizializzazione e all'ordine dei dati non è stata stimata. Gli intervalli di
confidenza riportati coprono soltanto il campionamento dei test set e sono perciò
ottimistici. Il rilievo pesa poco sul divario di generalizzazione, che vale oltre
venticinque punti, ma pesa molto sul confronto fra i due divari, dove la
differenza in gioco è di 1,74 punti contro un intervallo di ±1,93.

*Esplorazione parziale delle configurazioni*. Per Xception sono state provate due
profondità di scongelamento, il 24,6 e il 99,5 per cento dei parametri, senza le
vie di mezzo, quindi non si sa dove stia il punto migliore fra le due. Il
fine-tuning completo ha inoltre usato lo stesso tasso di apprendimento di quello
parziale, scelta voluta per isolare una sola variabile ma non ottimale, dato che
per riaddestrare una rete intera si usa di norma un valore più basso. Il suo
risultato è quindi semmai una sottostima. Il batch size, infine, è stato
confrontato solo sulla CNN custom.

*Etichette di FANE non verificate*. L'ispezione visiva ha fatto emergere alcune
etichette discutibili, ma non è stata condotta alcuna verifica sistematica. Non è
quindi possibile stabilire quanta parte del calo dipenda dal domain shift e quanta
dal rumore nelle annotazioni. È il limite che pesa di più sul risultato
principale, che poggia per giunta su una sola coppia di dataset.

= Conclusioni e sviluppi futuri

Il lavoro ha messo a confronto due approcci al riconoscimento delle espressioni
facciali, una rete convoluzionale residuale addestrata da zero e una Xception
pre-addestrata su ImageNet adattata tramite fine-tuning, valutandoli sia su
FER-2013 sia su FANE. Le quattro domande poste all'inizio trovano ora una
risposta.

Alla prima, se una rete addestrata da zero possa competere con il transfer
learning, la risposta è che dipende da come il transfer learning viene fatto. La
CNN custom raggiunge il 61,12 per cento di accuracy sul test set di FER-2013 e
batte di otto punti la versione di Xception con il solo ultimo blocco
scongelato, ma resta sotto di 4,74 punti alla versione con la rete interamente
riaddestrata, che arriva al 65,85 per cento. Il vantaggio del fine-tuning
completo si conserva anche fuori dominio, con 37,62 contro 34,62 per cento. Una
rete piccola e progettata sul problema è quindi competitiva, ma non batte un
modello grande usato bene.

Alla seconda domanda, quanto convenga riaddestrare, la risposta è netta ed è il
risultato più utile del lavoro. Passando dal 24,6 al 99,5 per cento dei parametri
addestrabili l'accuracy su FER-2013 sale di tredici punti e la macro-F1 da 0,4697
a 0,6210, a parità di architettura, pesi iniziali e tasso di apprendimento. Il
timore di rovinare i pesi pre-addestrati riaddestrandoli su un dataset piccolo
non ha trovato riscontro, mentre l'ha trovato il problema opposto, cioè che
congelarne troppi impedisce al modello di adattarsi. La fase di warm-up, provata
per proteggere i pesi pre-addestrati dai gradienti della testa casuale, non ha
invece prodotto alcuna differenza misurabile.

Alla terza domanda, quanto si perda cambiando dataset, la risposta è che si perde
moltissimo. La CNN custom scende di 26,50 punti percentuali di accuracy e
Xception di 28,23, e in macro-F1 entrambi lasciano per strada oltre il quaranta
per cento del proprio valore. Sono numeri che ridimensionano parecchio la fiducia
che si può riporre in un'accuracy riportata su un singolo dataset, e vanno letti
tenendo presente che una parte non quantificata di questo calo potrebbe dipendere
dal rumore nelle etichette di FANE piuttosto che dal domain shift.

Alla quarta domanda, se il pre-addestramento su ImageNet renda il modello più
robusto al cambio di dominio, la risposta è negativa. La differenza fra i due
divari di accuracy vale 1,74 punti con un intervallo di confidenza di ±1,93,
quindi non distinguibile da zero, e in termini relativi Xception perde il 45,4 per
cento della propria macro-F1 contro il 43,5 dell'altro modello, risultando semmai
il meno robusto dei due. Il pre-addestramento alza il livello assoluto su entrambi
i dataset senza ridurre la frazione che si perde attraversando il cambio di
dominio. Il calo sembra quindi dipendere dalle differenze fra i due dataset più
che dall'architettura o dal modo in cui i pesi sono stati inizializzati, e il
fatto che i due modelli sbaglino sulle stesse classi, con una correlazione di
0,989 fra i rispettivi recall su FANE, lo conferma.

A queste risposte se ne aggiunge una quinta, non prevista in partenza. Per tutta
la durata dell'addestramento lo scarto fra training e validation della CNN custom
è rimasto negativo, quindi la diagnostica abituale dell'overfitting non segnalava
alcun problema, eppure il modello si stava specializzando sulle caratteristiche di
FER-2013 al punto da perdere ventisei punti su un dataset diverso. Il controllo
fra training e validation non basta a rilevare questo tipo di sovradattamento,
perché i due insiemi condividono le stesse caratteristiche di acquisizione, e
serve un secondo dataset raccolto in modo indipendente.

Gli sviluppi possibili si dividono in due gruppi. Il primo comprende gli
interventi che consoliderebbero i risultati senza cambiare impostazione.
Ripetere ogni addestramento con seed diversi permetterebbe di affiancare a ogni
valore una stima della variabilità, cosa che renderebbe più solido soprattutto il
confronto fra i due divari, dove la differenza in gioco è piccola. Provare le
profondità intermedie di scongelamento, per esempio dal blocco dieci e dal blocco
tredici, direbbe se il guadagno cresca con gradualità o se esista una soglia oltre
la quale il modello si sblocca. Riaddestrare la rete intera con un tasso di
apprendimento più basso verificherebbe infine se il fine-tuning completo possa
fare ancora meglio, dato che qui ha usato lo stesso valore della versione
parziale.

Il secondo gruppo comprende le direzioni che estenderebbero il lavoro. La più
immediata è il class weighting, che permetterebbe di misurare quanto del crollo su
disgust dipenda dal prior appreso e quanto dalla reale difficoltà della classe.
Una seconda riguarda la risoluzione, perché resta da capire se il tetto intorno al
valore degli annotatori umani dipenda dai 48 per 48 pixel di FER-2013 o dal
problema in sé, e basterebbe ripetere il confronto su un dataset a risoluzione
maggiore. Un terzo filone consiste nell'addestrare su una combinazione dei due
dataset invece che su uno solo, per verificare se l'esposizione a immagini
eterogenee durante il training riduca il divario. La direzione più ambiziosa resta
l'applicazione di tecniche di domain adaptation, che affrontano il problema in
modo esplicito cercando di allineare le rappresentazioni apprese sui due domini, e
che alla luce di questi risultati sembrano più promettenti del semplice ricorso a
un modello pre-addestrato.
