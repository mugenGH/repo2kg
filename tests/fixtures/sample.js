// Sample JavaScript fixture for repo2kg tests

import { helper } from './helper';
const fs = require('fs');

function topLevelFunc(x, y) {
    return x + y;
}

const arrowFunc = (name) => {
    return helper(name);
};

class SampleClass {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return `Hello, ${this.name}`;
    }

    process(data) {
        return topLevelFunc(data.length, 1);
    }
}

export { topLevelFunc, arrowFunc, SampleClass };
