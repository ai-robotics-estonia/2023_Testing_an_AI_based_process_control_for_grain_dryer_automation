# Testing an AI based process control for grain dryer automation

## Summary
| Company Name | [Intellidry OÜ](https://intellidry.eu) |
| :--- | :--- |
| Development Team Lead Name | Veiko Vunder|
| Development Team Lead E-mail | [veiko.vunder@ut.ee](mailto:veiko.vunder@ut.ee) |
| Duration of the Demonstration Project | 06/2023-04/2024 |
| Final Report | [IntelliDryProjectReport2024.pdf](https://github.com/ai-robotics-estonia/Testing_an_AI_based_process_control_for_grain_dryer_automation/blob/main/assets/IntelliDryProjectReport2024.pdf) |

# Description
## Objectives of the Demonstration Project

The demo project aimed to develop an AI-based model solution for optimizing grain drying processes, designed to integrate with existing grain dryers. Currently, there are two main dryer types: batch dryers and continuous dryers. IntelliDry’s initial goal is to implement AI-driven drying control specifically for batch dryers, regardless of their current remote management capabilities. While we already offer remote management functionality, and many newer dryers support it, only a limited number are equipped with this feature. Through this demo project, we will create a baseline model that estimates grain moisture content - an essential parameter to reach IntelliDry's long-term goal of providing energy efficient heating schedule recommendations based on dryer type and other input data.

## Activities and Results of the Demonstration Project
### Challenge

The primary challenge is to develop a model that accurately describes the grain drying process. During the demo project, we assessed which data could be leveraged to build this model. The demo dryer provided temperature data for both incoming and outgoing air, as well as temperature readings from within the drying sections. An effective model would enable accurate predictions of drying time and operator intervention needs, while also helping to prevent over-drying. Although the initial plan was to control the drying process through an automated system, this approach was deferred. Instead, project resources were directed toward data collection, preparation, and model testing. Operating the dryer’s burner directly was deemed impractical, as it would detract from the project’s focus. While the potential for automated dryer control remains, this project concentrated on identifying key data requirements for the model and advancing its development.

### Data Sources

The technological solution includes a sensor set integrated with the dryer to track temperature and humidity changes during drying. This data is essential for developing the drying process model, a task that will continue beyond this demo project. We also manually measured grain and legume moisture throughout drying. Data collected includes:

- Incoming and outgoing air temperature and humidity
- Furnace air temperature
- Temperatures at the beginning, middle, and end of the drying section

Relevant scientific literature was reviewed to guide data analysis and model decisions.

### AI Technologies
In this demo project, we developed a physical model to capture essential elements of grain drying, including heat transfer from incoming air to grain, heat loss to the environment, and the mass of grain being dried. To estimate moisture content, we used a differential thin-layer drying model, where moisture change depends on the difference between current and equilibrium moisture. The equilibrium moisture content is in turn closely tied to grain temperature and well studied in the literature. Due to the complexity of solving these equations analytically, we applied numerical methods(fourth order Runge-Kutta), and optimized parameters using the Differential Evolution algorithm with data from the dryer. We also tested time-series machine learning models to predict moisture changes based on temperature but found that a larger dataset was necessary for accuracy. Despite that the model is currently grounded in physical drying processes, it includes learning capabilities through re-optimization techniques, enabling it to adapt and improve as new drying cycles are introduced.

### Technological Results

The model’s effectiveness for drying grains and legumes relies on data that was insufficiently collected during the demo dryer’s operation. The data gathered served as the foundation for development. With no drying occurring in winter, we tested identified relationships using existing datasets, revealing a strong correlation between grain temperature and moisture content, allowing us to estimate the drying behavior based on temperature readings.

For accurate predictions, it's crucial to consider temperature changes, influenced by unmeasured factors like flow rate, air velocity, and reservoir volume. The first two can be adjusted by the operator, impacting model parameters. Automating flow rate detection through temperature data will be a key focus of the next demo project.

During the drying period, operators monitored the lower grain temperature to estimate dryness. Based on the grain type, they determined the optimal time for moisture measurement and assessed batch readiness. A temperature-dependent cooling process was implemented and tested, with the control system initiating cooling upon reaching the operator-set grain temperature, thereby minimizing energy consumption and equipment wear.

### Technical Architecture

![technical-architecture](https://github.com/ai-robotics-estonia/Testing_an_AI_based_process_control_for_grain_dryer_automation/blob/main/assets/architecture.png)


### User Interface 
The user interface is available on both the server and client sides, with enhancements made to the operator’s interface. Operators can monitor dryer temperatures and control electric motors, with their statuses clearly displayed. For instance, if the burner malfunctions, the interface indicates both the motor status and the incoming air temperature. Sensor temperature readings are presented graphically, allowing operators to view cycles of up to six hours.

Operators can set a trigger value for one temperature sensor, which automatically initiates cooling for the batch. In the demo dryer, it was determined that the moisture level of oats is optimal when the lower temperature sensor approaches 55°C. The system can automatically initiate cooling upon reaching this threshold and notify the operator accordingly.

### Future Potential of the Technical Solution
The results and conclusions from the first project indicated that to improve the model, more training data is needed. Additional data is required from a greater variety of dryers. Furthermore, it remains unclear which model would perform best. Consequently, we decided to seek further support from AIRE for the AI model's development. With assistance from PRIA, we equipped ten additional dryers with sensors to collect data during the 2024 drying season. Development work is currently underway in the machine learning domain, and the company is also working on pricing and business models to launch the service for Estonian farmers.

### Lessons Learned
The technological solution successfully achieved one of its key objectives: enabling mobile monitoring and operation of the dryer. A baseline model was developed, which can fit the limited amount of data we have collected so far. However, this data has raised several questions that require additional information from other dryers for further refinement. By comparing the grain drying processes across different dryers, we can enhance the model by adding or adjusting parameters based on broader insights.

A significant lesson learned is the necessity of measuring a wider range of physical attributes for model training and development than initially anticipated. Although grain drying may appear straightforward based on initial analysis, the complexity of the process indicates that capturing more variables from various sources will improve the model's performance. Ultimately, the goal is to identify the minimum necessary hardware to enhance the model's predictive accuracy and ensure effective service delivery for Estonian farmers.

