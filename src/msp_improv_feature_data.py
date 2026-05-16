from src.iemocap_feature_data import IEMOCAPFeatureData


class MSPIMPROVFeatureData(IEMOCAPFeatureData):
    DATASET_GLOB = "MSP_IMPROV"

    def _resolve_pkl_path(self, data_path, split_name, l_type=None, a_type=None, v_type=None):
        import glob
        import os

        preferred_tokens = []
        for token in [l_type, a_type, v_type]:
            if token is not None:
                preferred_tokens.append(str(token).lower())

        primary_pattern = os.path.join(data_path, f"{self.DATASET_GLOB}_*_{split_name}.pkl")
        fallback_pattern = os.path.join(data_path, f"*{split_name}.pkl")
        candidates = sorted(set(glob.glob(primary_pattern) + glob.glob(fallback_pattern)))

        if not candidates:
            raise FileNotFoundError(
                f"Cannot find extracted MSP-IMPROV feature file for split '{split_name}' under {data_path}"
            )
        if len(candidates) == 1:
            return candidates[0]

        if preferred_tokens:
            scored = []
            for path in candidates:
                name = os.path.basename(path).lower()
                score = sum(int(token in name) for token in preferred_tokens)
                scored.append((score, path))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if scored[0][0] > 0:
                return scored[0][1]

        return candidates[0]
