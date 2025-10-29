The construction industry is a significant contributor to global greenhouse gas emissions, primarily due to the widespread use of Portland cement-based concrete. Cement production alone is estimated to account for roughly 5–8% of worldwide CO₂ emissions, making it one of the most energy-intensive and environmentally taxing industries.

In response to the urgent need for sustainable alternatives, geopolymer concrete has emerged as a promising solution. Unlike Portland cement concrete, geopolymer concrete replaces cement with industrial byproducts such as fly ash, slag, or metakaolin. These alternatives drastically reduce carbon emissions while maintaining, or in some cases even improving, the structural performance of concrete.

This project seeks to leverage machine learning (ML) techniques to predict the performance of metakaolin-based geopolymer concrete. Specifically, it investigates whether early-age parameters—initial setting time (IST) and final setting time (FST)—can serve as reliable predictors of two key mechanical properties: compressive strength (CS) and flexural strength (FS). By doing so, this project provides a practical and scalable framework to evaluate concrete without extensive laboratory testing.
















Interpret Results

Example

· If Raw Material A has r = -0.80 with CST, → it strongly reduces compressive strength when its % increases.

· If Raw Material B has r = +0.65 with CST, → it strongly increases compressive strength when its % increases.

· If Raw Material C has r = -0.10 with IST, → it has little effect on Initial Setting Time.


strong: |r| ≥ 0.70
moderate: 0.30 ≤ |r| < 0.70
weak: |r| < 0.30
------------------------------------------------------------------
increases: r > 0 → as the input goes up, the output tends to go up.
reduces: r < 0 → as the input goes up, the output tends to go down.
Near zero → r ≈ 0: little to no linear relationship.


