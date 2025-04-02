import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

// Define our tabs for different demonstrations
const TABS = {
  INTRO: 'Introduction',
  TENSORS: 'Tensors',
  AUTOGRAD: 'Automatic Differentiation',
  OPTIMIZATION: 'Gradient Descent',
  DATASETS: 'Datasets',
  NEURAL_NET: 'Neural Networks',
  GPU: 'GPU Acceleration'
};

const DeepLearningVisualizer = () => {
  const [activeTab, setActiveTab] = useState(TABS.INTRO);
  
  // Tensor Visualization State
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [tensorDims, setTensorDims] = useState([2, 2]);
  const [tensor1, setTensor1] = useState([[1, 2], [3, 4]]);
  const [tensor2, setTensor2] = useState([[5, 6], [7, 8]]);
  const [tensorOp, setTensorOp] = useState('add');
  const [tensorResult, setTensorResult] = useState([[6, 8], [10, 12]]);
  
  // Automatic Differentiation State
  const [xValue, setXValue] = useState(-3);
  const [functionType, setFunctionType] = useState('quadratic');
  const [autogradResult, setAutogradResult] = useState({
    xValue: -3,
    functionValue: 25,
    gradient: -10
  });
  
  // Gradient Descent State
  const [startPoint, setStartPoint] = useState(-3);
  const [learningRate, setLearningRate] = useState(0.1);
  const [optimizationSteps, setOptimizationSteps] = useState(10);
  const [gradientPath, setGradientPath] = useState([
    { x: -3, y: 25 },
    { x: -2.0, y: 16 },
    { x: -1.0, y: 9 },
    { x: 0.0, y: 4 },
    { x: 1.0, y: 1 },
    { x: 1.6, y: 0.16 },
    { x: 1.92, y: 0.0064 },
    { x: 1.984, y: 0.000256 },
    { x: 1.9968, y: 0.00001024 },
    { x: 1.99936, y: 4.0960000000000003e-7 }
  ]);
  
  // Datasets State
  const [datasetPoints, setDatasetPoints] = useState(100);
  const [datasetNoise, setDatasetNoise] = useState(0.2);
  const [batchSize, setBatchSize] = useState(10);
  const [datasetData, setDatasetData] = useState(
    Array.from({ length: 100 }, (_, i) => {
      const x = -3 + i * 6 / 99;
      const y = Math.sin(x) + (Math.random() - 0.5) * 0.4;
      return { x, y, type: Math.random() > 0.2 ? 'train' : 'test' };
    })
  );
  
  // Neural Networks State
  const [networkLayers, setNetworkLayers] = useState([2, 4, 1]);
  const [epochs, setEpochs] = useState(10);
  const [trainingLoss, setTrainingLoss] = useState([
    { epoch: 1, loss: 0.8 },
    { epoch: 2, loss: 0.6 },
    { epoch: 3, loss: 0.5 },
    { epoch: 4, loss: 0.3 },
    { epoch: 5, loss: 0.25 },
    { epoch: 6, loss: 0.2 },
    { epoch: 7, loss: 0.18 },
    { epoch: 8, loss: 0.15 },
    { epoch: 9, loss: 0.13 },
    { epoch: 10, loss: 0.12 }
  ]);
  
  // GPU Acceleration State
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [matrixSizes, setMatrixSizes] = useState([10, 50, 100, 500, 1000]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [gpuSpeedups, setGpuSpeedups] = useState([
    { size: 10, cpu: 0.001, gpu: 0.002, speedup: 0.5 },
    { size: 50, cpu: 0.01, gpu: 0.008, speedup: 1.25 },
    { size: 100, cpu: 0.05, gpu: 0.02, speedup: 2.5 },
    { size: 500, cpu: 0.5, gpu: 0.1, speedup: 5 },
    { size: 1000, cpu: 3.0, gpu: 0.3, speedup: 10 }
  ]);
  
  // Function to generate quadratic function data
  const generateFunctionData = (xMin=-5, xMax=5, numPoints=100) => {
    const data = [];
    
    for (let i = 0; i < numPoints; i++) {
      const x = xMin + (i * (xMax - xMin) / (numPoints - 1));
      let y;
      
      if (functionType === 'quadratic') {
        y = (x - 2) ** 2;  // f(x) = (x - 2)²
      } else if (functionType === 'sin') {
        y = Math.sin(x);  // f(x) = sin(x)
      } else if (functionType === 'exp') {
        y = Math.exp(x);  // f(x) = e^x
      }
      
      data.push({ x, y });
    }
    
    return data;
  };
  
  // Function to calculate the derivative of our functions
  const calculateDerivative = (x: number): number => {
    if (functionType === 'quadratic') {
      return 2 * (x - 2);  // f'(x) = 2(x - 2)
    } else if (functionType === 'sin') {
      return Math.cos(x);  // f'(x) = cos(x)
    } else if (functionType === 'exp') {
      return Math.exp(x);  // f'(x) = e^x
    }
    return 0;
  };
  
  // Function to generate gradient descent path
  const calculateGradientPath = (xStart: number, lr: number, steps: number): Array<{x: number, y: number}> => {
    const path: Array<{x: number, y: number}> = [];
    let x = xStart;
    
    for (let i = 0; i < steps; i++) {
      let y: number;
      if (functionType === 'quadratic') {
        y = (x - 2) ** 2;
      } else if (functionType === 'sin') {
        y = Math.sin(x);
      } else if (functionType === 'exp') {
        y = Math.exp(x);
      } else {
        y = 0; // Default fallback
      }
      
      path.push({ x, y });
      
      // Update x using the gradient
      const grad = calculateDerivative(x);
      x = x - lr * grad;
    }
    
    return path;
  };
  
  // Handler for learning rate change
  const handleLearningRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newLr = parseFloat(e.target.value);
    setLearningRate(newLr);
    setGradientPath(calculateGradientPath(startPoint, newLr, optimizationSteps));
  };
  
  // Handler for starting point change
  const handleStartPointChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newStart = parseFloat(e.target.value);
    setStartPoint(newStart);
    setGradientPath(calculateGradientPath(newStart, learningRate, optimizationSteps));
  };
  
  // Handler for steps change
  const handleStepsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSteps = parseInt(e.target.value);
    setOptimizationSteps(newSteps);
    setGradientPath(calculateGradientPath(startPoint, learningRate, newSteps));
  };
  
  // Handler for function type change
  const handleFunctionTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = e.target.value;
    setFunctionType(newType);
    
    // Update autograd result
    const x = xValue;
    let functionValue: number = 0;
    
    if (newType === 'quadratic') {
      functionValue = (x - 2) ** 2;
    } else if (newType === 'sin') {
      functionValue = Math.sin(x);
    } else if (newType === 'exp') {
      functionValue = Math.exp(x);
    }
    
    setAutogradResult({
      xValue: x,
      functionValue,
      gradient: calculateDerivative(x)
    });
    
    // Update gradient path
    setGradientPath(calculateGradientPath(startPoint, learningRate, optimizationSteps));
  };
  
  // Handler for tensor operation change
  const handleTensorOpChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const op = e.target.value;
    setTensorOp(op);
    
    // Calculate result based on operation
    const result = [];
    for (let i = 0; i < tensor1.length; i++) {
      const row = [];
      for (let j = 0; j < tensor1[0].length; j++) {
        if (op === 'add') {
          row.push(tensor1[i][j] + tensor2[i][j]);
        } else if (op === 'multiply') {
          row.push(tensor1[i][j] * tensor2[i][j]);
        } else if (op === 'subtract') {
          row.push(tensor1[i][j] - tensor2[i][j]);
        }
      }
      result.push(row);
    }
    
    setTensorResult(result);
  };
  
  // Generate synthetic dataset
  const generateDataset = (points: number, noise: number) => {
    return Array.from({ length: points }, (_, i) => {
      const x = -3 + i * 6 / (points - 1);
      const y = Math.sin(x) + (Math.random() - 0.5) * noise * 2;
      return { x, y, type: Math.random() > 0.2 ? 'train' : 'test' };
    });
  };
  
  // Handler for dataset parameters change
  const handleDatasetParamsChange = () => {
    setDatasetData(generateDataset(datasetPoints, datasetNoise));
  };
  
  // Generate training loss data
  const generateTrainingLoss = (numEpochs: number) => {
    return Array.from({ length: numEpochs }, (_, i) => {
      // Start with higher loss, gradually decrease with some randomness
      const baseValue = 1 - (i / numEpochs) * 0.9;
      const randomness = 0.05 * Math.random();
      return { epoch: i + 1, loss: baseValue - randomness };
    });
  };
  
  // Handler for epochs change
  const handleEpochsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newEpochs = parseInt(e.target.value);
    setEpochs(newEpochs);
    setTrainingLoss(generateTrainingLoss(newEpochs));
  };

  // Render the Introduction tab content
  const renderIntroduction = () => {
    // Inline styles for components
    const styles: {[key: string]: React.CSSProperties} = {
      container: {
        padding: '1.5rem',
        backgroundColor: 'white',
        borderRadius: '0.5rem',
        boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
      },
      heading: {
        fontSize: '1.5rem',
        fontWeight: 'bold',
        marginBottom: '1rem'
      },
      paragraph: {
        marginBottom: '1rem'
      },
      subheading: {
        fontSize: '1.25rem',
        fontWeight: '600',
        marginBottom: '0.5rem',
        marginTop: '1.5rem'
      },
      gridContainer: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '1rem',
        marginBottom: '1.5rem'
      },
      card: {
        padding: '1rem',
        borderRadius: '0.25rem'
      },
      blueCard: {
        backgroundColor: '#EBF5FF',
      },
      greenCard: {
        backgroundColor: '#F0FFF4',
      },
      yellowCard: {
        backgroundColor: '#FFFFF0',
      },
      purpleCard: {
        backgroundColor: '#FAF5FF',
      },
      redCard: {
        backgroundColor: '#FFF5F5',
      },
      indigoCard: {
        backgroundColor: '#F0F5FF',
      },
      cardTitle: {
        fontWeight: 'bold'
      },
      gettingStarted: {
        backgroundColor: '#F3F4F6',
        padding: '1rem',
        borderRadius: '0.5rem',
        marginTop: '1.5rem'
      }
    };

    return (
      <div style={styles.container}>
        <h2 style={styles.heading}>Welcome to Deep Learning Foundations</h2>
        
        <p style={styles.paragraph}>
          This interactive app demonstrates the core concepts from Chapter 1 of "Inside Deep Learning".
          You can explore each concept through visualizations and interactive controls.
        </p>
        
        <h3 style={styles.subheading}>What You'll Learn</h3>
        
        <div style={styles.gridContainer}>
          <div style={{...styles.card, ...styles.blueCard}}>
            <h4 style={styles.cardTitle}>Tensors</h4>
            <p>The fundamental data structure for deep learning</p>
          </div>
          
          <div style={{...styles.card, ...styles.greenCard}}>
            <h4 style={styles.cardTitle}>Automatic Differentiation</h4>
            <p>How machines learn by calculating gradients</p>
          </div>
          
          <div style={{...styles.card, ...styles.yellowCard}}>
            <h4 style={styles.cardTitle}>Gradient Descent</h4>
            <p>The optimization process that powers learning</p>
          </div>
          
          <div style={{...styles.card, ...styles.purpleCard}}>
            <h4 style={styles.cardTitle}>Datasets</h4>
            <p>How data is prepared and fed to neural networks</p>
          </div>
          
          <div style={{...styles.card, ...styles.redCard}}>
            <h4 style={styles.cardTitle}>Neural Networks</h4>
            <p>Building and training simple neural networks</p>
          </div>
          
          <div style={{...styles.card, ...styles.indigoCard}}>
            <h4 style={styles.cardTitle}>GPU Acceleration</h4>
            <p>How hardware speeds up deep learning computations</p>
          </div>
        </div>
        
        <p style={styles.paragraph}>
          Select any tab above to start exploring these concepts. Each section includes 
          interactive controls that let you experiment with the parameters and see the results in real-time.
        </p>
        
        <div style={styles.gettingStarted}>
          <h3 style={{fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem'}}>Getting Started</h3>
          <p>
            Click on the "Tensors" tab to begin your journey into deep learning fundamentals.
            Make sure to try the interactive elements on each page!
          </p>
        </div>
      </div>
    );
  };

  // Render the Tensors tab content
  const renderTensors = () => {
    // Inline styles for the Tensors section
    const styles: {[key: string]: React.CSSProperties} = {
      container: {
        padding: '1.5rem',
        backgroundColor: 'white',
        borderRadius: '0.5rem',
        boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
      },
      heading: {
        fontSize: '1.5rem',
        fontWeight: 'bold',
        marginBottom: '1rem'
      },
      paragraph: {
        marginBottom: '1.5rem'
      },
      grid: {
        display: 'grid',
        gridTemplateColumns: '1fr',
        gap: '1.5rem',
        marginBottom: '1.5rem'
      },
      gridLarge: {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '1.5rem',
        marginBottom: '1.5rem'
      },
      tensorBox: {
        backgroundColor: '#F9FAFB',
        padding: '1rem',
        borderRadius: '0.5rem',
      },
      boxHeading: {
        fontSize: '1.125rem',
        fontWeight: '600',
        marginBottom: '0.5rem'
      },
      tensorGrid: {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '0.5rem',
        marginBottom: '1rem'
      },
      input: {
        padding: '0.5rem',
        border: '1px solid #D1D5DB',
        borderRadius: '0.25rem',
        textAlign: 'center'
      },
      select: {
        padding: '0.5rem',
        border: '1px solid #D1D5DB',
        borderRadius: '0.25rem',
        width: '100%'
      },
      resultBox: {
        backgroundColor: '#EBF5FF',
        padding: '1rem',
        borderRadius: '0.5rem',
        marginBottom: '1.5rem'
      },
      resultGrid: {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '0.5rem'
      },
      resultCell: {
        padding: '0.5rem',
        border: '1px solid #D1D5DB',
        backgroundColor: 'white',
        borderRadius: '0.25rem',
        textAlign: 'center'
      },
      keyPoints: {
        backgroundColor: '#F3F4F6',
        padding: '1rem',
        borderRadius: '0.5rem'
      },
      keyPointsList: {
        listStyleType: 'disc',
        marginLeft: '1.25rem',
      },
      keyPointsItem: {
        marginBottom: '0.25rem'
      }
    };

    return (
      <div style={styles.container}>
        <h2 style={styles.heading}>Tensors: The Building Blocks</h2>
        
        <p style={styles.paragraph}>
          Tensors are multi-dimensional arrays that form the foundation of deep learning.
          They can represent scalars (0D), vectors (1D), matrices (2D), and higher-dimensional data.
        </p>
        
        <div style={styles.gridLarge}>
          <div style={styles.tensorBox}>
            <h3 style={styles.boxHeading}>Tensor 1</h3>
            <div style={styles.tensorGrid}>
              {tensor1.map((row, i) => (
                row.map((value, j) => (
                  <input
                    key={`t1-${i}-${j}`}
                    type="number"
                    style={styles.input}
                    value={value}
                    onChange={(e) => {
                      const newTensor = [...tensor1];
                      newTensor[i][j] = parseFloat(e.target.value);
                      setTensor1(newTensor);
                      handleTensorOpChange({ target: { value: tensorOp } } as React.ChangeEvent<HTMLSelectElement>);
                    }}
                  />
                ))
              ))}
            </div>
          </div>
          
          <div style={styles.tensorBox}>
            <h3 style={styles.boxHeading}>Tensor 2</h3>
            <div style={styles.tensorGrid}>
              {tensor2.map((row, i) => (
                row.map((value, j) => (
                  <input
                    key={`t2-${i}-${j}`}
                    type="number"
                    style={styles.input}
                    value={value}
                    onChange={(e) => {
                      const newTensor = [...tensor2];
                      newTensor[i][j] = parseFloat(e.target.value);
                      setTensor2(newTensor);
                      handleTensorOpChange({ target: { value: tensorOp } } as React.ChangeEvent<HTMLSelectElement>);
                    }}
                  />
                ))
              ))}
            </div>
          </div>
        </div>
        
        <div style={{marginBottom: '1.5rem'}}>
          <h3 style={styles.boxHeading}>Operation</h3>
          <select
            style={styles.select}
            value={tensorOp}
            onChange={handleTensorOpChange}
          >
            <option value="add">Addition</option>
            <option value="subtract">Subtraction</option>
            <option value="multiply">Element-wise Multiplication</option>
          </select>
        </div>
        
        <div style={styles.resultBox}>
          <h3 style={styles.boxHeading}>Result</h3>
          <div style={styles.resultGrid}>
            {tensorResult.map((row, i) => (
              row.map((value, j) => (
                <div key={`result-${i}-${j}`} style={styles.resultCell}>
                  {value}
                </div>
              ))
            ))}
          </div>
        </div>
        
        <div style={styles.keyPoints}>
          <h3 style={styles.boxHeading}>Key Points</h3>
          <ul style={styles.keyPointsList}>
            <li style={styles.keyPointsItem}>Tensors are the primary data structure in deep learning frameworks</li>
            <li style={styles.keyPointsItem}>PyTorch and TensorFlow operations work with tensors</li>
            <li style={styles.keyPointsItem}>Tensors can represent inputs, outputs, and model parameters</li>
            <li style={styles.keyPointsItem}>GPUs are optimized for tensor operations, making deep learning faster</li>
          </ul>
        </div>
      </div>
    );
  };

  // Render the Automatic Differentiation tab content
  const renderAutoDiff = () => (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">Automatic Differentiation</h2>
      
      <p className="mb-6">
        Automatic differentiation is the engine that powers learning in neural networks.
        It efficiently computes gradients (derivatives) needed for optimization.
      </p>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="bg-gray-50 p-4 rounded-lg col-span-1">
          <h3 className="text-lg font-semibold mb-2">Function Settings</h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Function Type</label>
            <select
              className="p-2 border rounded w-full"
              value={functionType}
              onChange={handleFunctionTypeChange}
            >
              <option value="quadratic">Quadratic: f(x) = (x - 2)²</option>
              <option value="sin">Sine: f(x) = sin(x)</option>
              <option value="exp">Exponential: f(x) = e^x</option>
            </select>
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">X Value: {xValue}</label>
            <input
              type="range"
              min="-5"
              max="5"
              step="0.1"
              value={xValue}
              onChange={(e) => {
                const newX = parseFloat(e.target.value);
                setXValue(newX);
                
                let functionValue: number = 0;
                if (functionType === 'quadratic') {
                  functionValue = (newX - 2) ** 2;
                } else if (functionType === 'sin') {
                  functionValue = Math.sin(newX);
                } else if (functionType === 'exp') {
                  functionValue = Math.exp(newX);
                }
                
                setAutogradResult({
                  xValue: newX,
                  functionValue,
                  gradient: calculateDerivative(newX)
                });
              }}
              className="w-full"
            />
          </div>
          
          <div className="bg-blue-50 p-3 rounded-lg">
            <h4 className="font-semibold mb-1">Results at x = {xValue.toFixed(2)}</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="font-medium">Function Value:</div>
              <div>{autogradResult.functionValue.toFixed(4)}</div>
              
              <div className="font-medium">Gradient (df/dx):</div>
              <div>{autogradResult.gradient.toFixed(4)}</div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg col-span-2">
          <h3 className="text-lg font-semibold mb-2">Function and Derivative</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={generateFunctionData()}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="y" 
                name="f(x)" 
                stroke="#8884d8" 
                dot={false}
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                data={generateFunctionData().map(point => ({
                  x: point.x,
                  y: calculateDerivative(point.x)
                }))} 
                dataKey="y" 
                name="f'(x)" 
                stroke="#82ca9d" 
                dot={false}
                strokeWidth={2}
              />
              <Line 
                type="monotone" 
                data={[{x: xValue, y: 0}]} 
                dataKey="y" 
                name="Current X" 
                stroke="red" 
                strokeWidth={0}
                dot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Key Points</h3>
        <ul className="list-disc ml-5 space-y-1">
          <li>Automatic differentiation computes exact derivatives efficiently</li>
          <li>PyTorch tracks operations to calculate gradients through a computation graph</li>
          <li>Gradients tell us how to adjust parameters to minimize loss functions</li>
          <li>The sign of the gradient indicates the direction to move for minimization</li>
          <li>The magnitude of the gradient indicates how steep the function is at that point</li>
        </ul>
      </div>
    </div>
  );

  // Render the Gradient Descent tab content
  const renderOptimization = () => (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">Gradient Descent Optimization</h2>
      
      <p className="mb-6">
        Gradient descent is the optimization algorithm that powers neural network training.
        It uses gradients to iteratively update parameters and minimize loss functions.
      </p>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="bg-gray-50 p-4 rounded-lg col-span-1">
          <h3 className="text-lg font-semibold mb-2">Optimization Settings</h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Function Type</label>
            <select
              className="p-2 border rounded w-full"
              value={functionType}
              onChange={handleFunctionTypeChange}
            >
              <option value="quadratic">Quadratic: f(x) = (x - 2)²</option>
              <option value="sin">Sine: f(x) = sin(x)</option>
              <option value="exp">Exponential: f(x) = e^x</option>
            </select>
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Starting Point: {startPoint}
            </label>
            <input
              type="range"
              min="-5"
              max="5"
              step="0.1"
              value={startPoint}
              onChange={handleStartPointChange}
              className="w-full"
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Learning Rate: {learningRate}
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={learningRate}
              onChange={handleLearningRateChange}
              className="w-full"
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Optimization Steps: {optimizationSteps}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              step="1"
              value={optimizationSteps}
              onChange={handleStepsChange}
              className="w-full"
            />
          </div>
          
          <div className="bg-blue-50 p-3 rounded-lg">
            <h4 className="font-semibold mb-1">Optimization Results</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="font-medium">Starting X:</div>
              <div>{startPoint.toFixed(2)}</div>
              
              <div className="font-medium">Starting Value:</div>
              <div>{gradientPath[0]?.y.toFixed(4)}</div>
              
              <div className="font-medium">Final X:</div>
              <div>{gradientPath[gradientPath.length-1]?.x.toFixed(4)}</div>
              
              <div className="font-medium">Final Value:</div>
              <div>{gradientPath[gradientPath.length-1]?.y.toFixed(6)}</div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg col-span-2">
          <h3 className="text-lg font-semibold mb-2">Gradient Descent Visualization</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" domain={[-5, 5]} />
              <YAxis domain={[0, 'dataMax']} />
              <Tooltip />
              <Legend />
              
              {/* The function curve */}
              <Line 
                data={generateFunctionData()} 
                type="monotone" 
                dataKey="y" 
                name="Function f(x)" 
                stroke="#8884d8" 
                dot={false}
              />
              
              {/* The gradient descent path */}
              <Line 
                data={gradientPath} 
                type="monotone" 
                dataKey="y" 
                name="Optimization Path" 
                stroke="#ff0000" 
                strokeWidth={2}
                dot={{ stroke: '#ff0000', strokeWidth: 2, r: 4 }} 
              />
              
              {/* Starting point */}
              <Line 
                data={[gradientPath[0]]} 
                type="monotone" 
                dataKey="y" 
                name="Starting Point" 
                stroke="#00ff00"
                strokeWidth={0}
                dot={{ stroke: '#00ff00', strokeWidth: 2, r: 8 }} 
              />
              
              {/* Ending point */}
              <Line 
                data={[gradientPath[gradientPath.length-1]]} 
                type="monotone" 
                dataKey="y" 
                name="Final Point" 
                stroke="#0000ff"
                strokeWidth={0}
                dot={{ stroke: '#0000ff', strokeWidth: 2, r: 8 }} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Key Points</h3>
        <ul className="list-disc ml-5 space-y-1">
          <li>Gradient descent uses derivatives to find the minimum of a function</li>
          <li>The learning rate controls how large each step is</li>
          <li>If learning rate is too small: slow convergence</li>
          <li>If learning rate is too large: may overshoot or diverge</li>
          <li>Neural networks use this same process to minimize loss functions</li>
          <li>Finding the minimum means the model has learned optimal parameters</li>
        </ul>
      </div>
    </div>
  );

  // Render the Datasets tab content
  const renderDatasets = () => (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">Datasets and Data Loading</h2>
      
      <p className="mb-6">
        Neural networks learn from data. The Dataset and DataLoader abstractions help
        manage data efficiently during training.
      </p>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="bg-gray-50 p-4 rounded-lg col-span-1">
          <h3 className="text-lg font-semibold mb-2">Dataset Settings</h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Number of Data Points: {datasetPoints}
            </label>
            <input
              type="range"
              min="10"
              max="200"
              step="10"
              value={datasetPoints}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDatasetPoints(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Noise Level: {datasetNoise.toFixed(1)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={datasetNoise}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setDatasetNoise(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Batch Size: {batchSize}
            </label>
            <input
              type="range"
              min="1"
              max="50"
              step="1"
              value={batchSize}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBatchSize(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          
          <button
            className="mt-2 bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
            onClick={handleDatasetParamsChange}
          >
            Generate New Dataset
          </button>
          
          <div className="bg-blue-50 p-3 rounded-lg mt-4">
            <h4 className="font-semibold mb-1">Dataset Information</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="font-medium">Total Points:</div>
              <div>{datasetData.length}</div>
              
              <div className="font-medium">Training Points:</div>
              <div>{datasetData.filter(d => d.type === 'train').length}</div>
              
              <div className="font-medium">Testing Points:</div>
              <div>{datasetData.filter(d => d.type === 'test').length}</div>
              
              <div className="font-medium">Batches:</div>
              <div>{Math.ceil(datasetData.length / batchSize)}</div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg col-span-2">
          <h3 className="text-lg font-semibold mb-2">Dataset Visualization</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="x" />
              <YAxis type="number" dataKey="y" name="y" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              
              {/* Training data */}
              <Scatter 
                name="Training Data" 
                data={datasetData.filter(d => d.type === 'train')} 
                fill="#8884d8" 
              />
              
              {/* Testing data */}
              <Scatter 
                name="Testing Data" 
                data={datasetData.filter(d => d.type === 'test')} 
                fill="#82ca9d" 
              />
              
              {/* True function */}
              <Line 
                type="monotone" 
                dataKey="y" 
                data={Array.from({ length: 100 }, (_, i) => {
                  const x = -3 + i * 6 / 99;
                  return { x, y: Math.sin(x) };
                })}
                name="True Function (sin(x))" 
                stroke="#ff7300" 
                dot={false}
              />
            </ScatterChart>
          </ResponsiveContainer>
          
          <div className="mt-4 p-3 bg-yellow-50 rounded-lg">
            <h4 className="font-semibold mb-1">Sample Batch</h4>
            <div className="grid grid-cols-5 gap-1">
              {Array.from({ length: Math.min(10, batchSize) }).map((_, i) => (
                <div key={i} className="p-2 bg-white border rounded text-center text-sm">
                  ({datasetData[i]?.x.toFixed(1)}, {datasetData[i]?.y.toFixed(1)})
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Key Points</h3>
        <ul className="list-disc ml-5 space-y-1">
          <li>The Dataset class defines what data is available and how to access it</li>
          <li>The DataLoader provides batched data and shuffling during training</li>
          <li>Batching improves training efficiency and convergence</li>
          <li>Data is split into training and testing sets to evaluate generalization</li>
          <li>Loading data on-demand reduces memory usage for large datasets</li>
        </ul>
      </div>
    </div>
  );

  // Render the Neural Networks tab content
  const renderNeuralNetworks = () => (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">Neural Networks</h2>
      
      <p className="mb-6">
        Neural networks are composed of layers of neurons that transform input data.
        They learn by updating weights through backpropagation and gradient descent.
      </p>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="bg-gray-50 p-4 rounded-lg col-span-1">
          <h3 className="text-lg font-semibold mb-2">Network Configuration</h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Network Architecture</label>
            <div className="flex items-center space-x-2">
              <input
                type="number"
                min="1"
                max="10"
                value={networkLayers[0]}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  const newLayers = [...networkLayers];
                  newLayers[0] = parseInt(e.target.value);
                  setNetworkLayers(newLayers);
                }}
                className="w-16 p-1 border rounded text-center"
              />
              <span className="text-gray-500">→</span>
              <input
                type="number"
                min="1"
                max="50"
                value={networkLayers[1]}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  const newLayers = [...networkLayers];
                  newLayers[1] = parseInt(e.target.value);
                  setNetworkLayers(newLayers);
                }}
                className="w-16 p-1 border rounded text-center"
              />
              <span className="text-gray-500">→</span>
              <input
                type="number"
                min="1"
                max="10"
                value={networkLayers[2]}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  const newLayers = [...networkLayers];
                  newLayers[2] = parseInt(e.target.value);
                  setNetworkLayers(newLayers);
                }}
                className="w-16 p-1 border rounded text-center"
              />
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Input → Hidden → Output
            </div>
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">
              Number of Epochs: {epochs}
            </label>
            <input
              type="range"
              min="1"
              max="50"
              step="1"
              value={epochs}
              onChange={handleEpochsChange}
              className="w-full"
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Learning Rate: 0.01</label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value="0.01"
              disabled
              className="w-full"
            />
          </div>
          
          <button
            className="mt-2 bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded w-full"
            onClick={() => setTrainingLoss(generateTrainingLoss(epochs))}
          >
            Train Model
          </button>
          
          <div className="bg-blue-50 p-3 rounded-lg mt-4">
            <h4 className="font-semibold mb-1">Network Information</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="font-medium">Total Layers:</div>
              <div>{networkLayers.length}</div>
              
              <div className="font-medium">Parameters:</div>
              <div>{(networkLayers[0] * networkLayers[1] + networkLayers[1] + networkLayers[1] * networkLayers[2] + networkLayers[2]).toLocaleString()}</div>
              
              <div className="font-medium">Final Loss:</div>
              <div>{trainingLoss[trainingLoss.length - 1]?.loss.toFixed(4)}</div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg col-span-2">
          <h3 className="text-lg font-semibold mb-2">Training Visualization</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={trainingLoss}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis domain={[0, 'dataMax']} />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="loss" 
                name="Training Loss" 
                stroke="#8884d8" 
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
          
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">Network Visualization</h3>
            <div className="flex justify-center items-center h-20 bg-white rounded border">
              <div className="flex items-center">
                {/* Input layer */}
                <div className="flex flex-col items-center mx-4">
                  <div className="text-xs mb-1">Input</div>
                  <div className="flex flex-col space-y-1">
                    {Array.from({ length: Math.min(5, networkLayers[0]) }).map((_, i) => (
                      <div key={i} className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs">
                        {i+1}
                      </div>
                    ))}
                    {networkLayers[0] > 5 && (
                      <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs">
                        ...
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Connection lines */}
                <div className="w-10 h-1 bg-gray-300"></div>
                
                {/* Hidden layer */}
                <div className="flex flex-col items-center mx-4">
                  <div className="text-xs mb-1">Hidden</div>
                  <div className="flex flex-col space-y-1">
                    {Array.from({ length: Math.min(5, networkLayers[1]) }).map((_, i) => (
                      <div key={i} className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white text-xs">
                        {i+1}
                      </div>
                    ))}
                    {networkLayers[1] > 5 && (
                      <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white text-xs">
                        ...
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Connection lines */}
                <div className="w-10 h-1 bg-gray-300"></div>
                
                {/* Output layer */}
                <div className="flex flex-col items-center mx-4">
                  <div className="text-xs mb-1">Output</div>
                  <div className="flex flex-col space-y-1">
                    {Array.from({ length: Math.min(5, networkLayers[2]) }).map((_, i) => (
                      <div key={i} className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center text-white text-xs">
                        {i+1}
                      </div>
                    ))}
                    {networkLayers[2] > 5 && (
                      <div className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center text-white text-xs">
                        ...
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Key Points</h3>
        <ul className="list-disc ml-5 space-y-1">
          <li>Neural networks are composed of layers of neurons</li>
          <li>Each neuron applies weights to inputs and passes through an activation function</li>
          <li>Networks learn by adjusting weights to minimize a loss function</li>
          <li>Backpropagation uses the chain rule to compute gradients efficiently</li>
          <li>Training typically happens in epochs (passes through the entire dataset)</li>
          <li>More complex networks can learn more complex functions but may overfit</li>
        </ul>
      </div>
    </div>
  );

  // Render the GPU Acceleration tab content
  const renderGPU = () => (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">GPU Acceleration</h2>
      
      <p className="mb-6">
        Graphics Processing Units (GPUs) dramatically speed up deep learning by performing
        many tensor operations in parallel. PyTorch makes it easy to use GPU acceleration.
      </p>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">GPU vs CPU Performance</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={gpuSpeedups}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="size" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="cpu" 
                name="CPU Time (seconds)" 
                stroke="#8884d8" 
                dot={{ r: 4 }}
              />
              <Line 
                type="monotone" 
                dataKey="gpu" 
                name="GPU Time (seconds)" 
                stroke="#82ca9d" 
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">GPU Speedup Factor</h3>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={gpuSpeedups}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="size" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="speedup" 
                name="Speedup Factor (CPU Time / GPU Time)" 
                stroke="#ff7300" 
                dot={{ r: 4 }}
              />
              <Line 
                type="monotone" 
                data={[
                  { size: 10, speedup: 1 },
                  { size: 1000, speedup: 1 }
                ]} 
                dataKey="speedup" 
                name="Break-even Line" 
                stroke="#ff0000" 
                strokeDasharray="5 5"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="bg-blue-50 p-4 rounded-lg mb-6">
        <h3 className="text-lg font-semibold mb-2">PyTorch GPU Code Example</h3>
        <pre className="bg-gray-800 text-white p-3 rounded overflow-x-auto">
          <code>{`# Move a tensor to GPU
tensor = torch.tensor([1, 2, 3])
tensor_gpu = tensor.to('cuda')  # moves to GPU

# Move back to CPU
tensor_cpu = tensor_gpu.cpu()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Create tensor directly on GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Operations run on GPU
z = x @ y  # matrix multiplication`}</code>
        </pre>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Key Points</h3>
        <ul className="list-disc ml-5 space-y-1">
          <li>GPUs are specialized for parallel computing, which is ideal for tensor operations</li>
          <li>Small operations may be slower on GPU due to data transfer overhead</li>
          <li>Large matrices and neural networks see dramatic speedups on GPUs</li>
          <li>PyTorch makes GPU usage simple with the .to(device) method</li>
          <li>All tensors in an operation must be on the same device (GPU or CPU)</li>
          <li>Modern deep learning would be impractical without GPU acceleration</li>
        </ul>
      </div>
    </div>
  );

  // Determine which content to render based on the active tab
  const renderContent = () => {
    switch (activeTab) {
      case TABS.INTRO:
        return renderIntroduction();
      case TABS.TENSORS:
        return renderTensors();
      case TABS.AUTOGRAD:
        return renderAutoDiff();
      case TABS.OPTIMIZATION:
        return renderOptimization();
      case TABS.DATASETS:
        return renderDatasets();
      case TABS.NEURAL_NET:
        return renderNeuralNetworks();
      case TABS.GPU:
        return renderGPU();
      default:
        return renderIntroduction();
    }
  };

  // Main component styles
  const mainStyles: {[key: string]: React.CSSProperties} = {
    container: {
      minHeight: '100vh',
      backgroundColor: '#F3F4F6',
      padding: '1.5rem',
    },
    content: {
      maxWidth: '72rem',
      margin: '0 auto',
    },
    header: {
      backgroundColor: 'white',
      boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
      borderRadius: '0.5rem',
      padding: '1rem',
      marginBottom: '1.5rem',
    },
    title: {
      fontSize: '1.875rem',
      fontWeight: 'bold',
      color: '#1F2937',
    },
    subtitle: {
      color: '#4B5563',
    },
    tabContainer: {
      display: 'flex',
      overflowX: 'auto' as const,
      backgroundColor: 'white',
      borderTopLeftRadius: '0.5rem',
      borderTopRightRadius: '0.5rem',
      boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
      marginBottom: '0.25rem',
      padding: '0.25rem',
    },
    tab: {
      padding: '0.5rem 1rem',
      fontSize: '0.875rem',
      fontWeight: '500',
      borderRadius: '0.5rem',
      marginRight: '0.25rem',
      whiteSpace: 'nowrap' as const,
      transition: 'all 0.2s',
      cursor: 'pointer',
    },
    activeTab: {
      backgroundColor: '#3B82F6',
      color: 'white',
    },
    inactiveTab: {
      backgroundColor: '#F3F4F6',
      color: '#4B5563',
    },
  };

  return (
    <div style={mainStyles.container}>
      <div style={mainStyles.content}>
        <header style={mainStyles.header}>
          <h1 style={mainStyles.title}>Deep Learning Foundations</h1>
          <p style={mainStyles.subtitle}>
            An interactive exploration of fundamental deep learning concepts
          </p>
        </header>
        
        {/* Navigation Tabs */}
        <div style={mainStyles.tabContainer}>
          {Object.values(TABS).map((tab) => (
            <button
              key={tab}
              style={{
                ...mainStyles.tab,
                ...(activeTab === tab ? mainStyles.activeTab : mainStyles.inactiveTab)
              }}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
        
        {/* Content Area */}
        {renderContent()}
      </div>
    </div>
  );
};

export default DeepLearningVisualizer;
