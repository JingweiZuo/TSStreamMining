## Incremental and Adaptive Feature Exploration over Time Series Stream
<p align="right">DAVID Lab <br/> Unversity of Versaille Saint-Quentin (UVSQ)<br/> Université Paris-Saclay</p>
By adopting the concepts Shapelet and Matrix Profile, we conduct the first attempt to extract the adaptive features from Time Series Stream, two streaming contexts are considered: 
- For data source with stable concept, learning model will be updated incrementally; 
- For data source with drifting concept, we extract the adaptive features under the most recent concept

### Main contributions:

1. A novel strategy to evaluate Shapelet, which shows the first attempt of transferring the techniques in Time Series community to Data Stream community 

   <img src="figures/Loss_Func_Plot.png" width="50%" height="50%" />  

   Figure 1. Shapelet Evaluation by a loss-smoothed  approach.

2. Test-then-Train Stategy: The novel strategy, not only accelerates the incremental Shapelet extraction in stable-concept context, but also helps with detecting Concept Drift in streaming context.

   <img src="figures/ISETS_Structure_BN.png" width="60%" height="60%" />

3. Elastic Caching Mechanism in Streaming context

   <img src="figures/Caching_mechanism_part.png" width="80%" height="80%" />

4. Scalability & Explainability 

5. Traceability of extracted features

